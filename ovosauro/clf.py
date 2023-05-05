import json
import os
import random
from itertools import chain, combinations
from threading import RLock

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from speech_recognition import Recognizer, AudioFile

from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier, SklearnOVOSVotingClassifier
from ovosauro import AlloSaurus


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [_ for _ in chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)) if len(_) > 1]


class OVOSauro:
    def __init__(self, langs, clf=None, pipeline="tfidf"):
        self.langs = langs
        clf = clf or [SVC(probability=True),
                      LogisticRegression(),
                      DecisionTreeClassifier()]

        if isinstance(clf, list):
            self.classifier = SklearnOVOSVotingClassifier(voter_clfs=clf, pipeline_id=pipeline)
        else:
            self.classifier = SklearnOVOSClassifier(pipeline_id=pipeline, pipeline_clf=clf)

        self.lang_features = {

        }
        self.lock = RLock()  # ensure no failures if intent registered during inference
        self.sauro = AlloSaurus()

    def recognize(self, audio_data, unknown=False):
        feats = [self.sauro.recognize(audio_data)]

        with self.lock:
            classes = self.classifier.clf.classes_
            probs = self.classifier.clf.predict_proba(feats)[0]
            a = list(zip(classes, probs))
            if not unknown:
                a = [_ for _ in a if _[0] != "other"]
                total = sum(_[1] for _ in a)
                a = [(_[0], _[1] / total) for _ in a]
            return a

    def load_features(self, folder):
        # folder of pre-processed features, see scripts/gather_features.py
        for lang in os.listdir(folder):
            if lang not in self.lang_features:
                self.lang_features[lang] = []
            p = f"{folder}/{lang}"
            for featfile in os.listdir(p):
                if not featfile.endswith(".txt"):
                    continue
                with open(f"{p}/{featfile}") as f:
                    try:
                        feats = f.read().strip()
                        self.lang_features[lang].append(feats)
                    # print("loaded ", featfile)
                    except:
                        continue
        print("loaded feats")

    def train(self, X, y):
        self.classifier.train(X, y)

    def load_data(self, unknown=True):
        data = []
        for lang, featlist in self.lang_features.items():
            if lang not in self.langs and not unknown:
                continue

            for f in featlist:
                if lang not in self.langs:
                    data.append((f, "other"))
                else:
                    data.append((f, lang))

        random.shuffle(data)
        X = []
        Y = []
        for x, y in data:
            X.append(x)
            Y.append(y)
        return X, Y

    def save(self, path):
        self.classifier.save(path)

    def load(self, path):
        self.classifier.load_from_file(path)


if __name__ == "__main__":
    hello = ["hello human", "hello there", "hey", "hello", "hi"]
    name = ["my name is {name}", "call me {name}", "I am {name}",
            "the name is {name}", "{name} is my name", "{name} is my name"]
    joke = ["tell me a joke", "say a joke", "tell joke"]
    scores = {}
    try:
        with open(f"/home/miro/PycharmProjects/ovosauro/pretrained/accuracy.json", "r") as _f:
            scores = json.load(_f)
    except:
        pass

    # single clf
    clf = SVC(probability=True)
    clf = [SVC(probability=True), LogisticRegression(), DecisionTreeClassifier()]

    # pre defined pipelines from ovos-classifiers
    clf_pipeline = "tfidf"  # "tfidf"
    lang_combos = powerset(sorted(["en", "pt", "fr", "it", "es", "de", "fi", "nl"]))
    random.shuffle(lang_combos)
    dt = None

    for langs in lang_combos:
        langs = list(langs)
        print(langs)
        if os.path.isfile(f"/home/miro/PycharmProjects/ovosauro/pretrained/voter_{'_'.join(langs)}.pkl"):
            continue

        engine = OVOSauro(langs, clf, clf_pipeline)
        feats = "/home/miro/PycharmProjects/ovosauro/scripts/feats"
        if dt is None:
            engine.load_features(feats)
            dt = engine.lang_features
        else:
            engine.lang_features = dt

        X, y = engine.load_data()
        n = int(len(X) * 0.7)
        print(n)

        Xtest = X[n:]
        X = X[:n]

        ytest = y[n:]
        y = y[:n]

        engine.train(X, y)
        score = engine.classifier.score(Xtest, ytest)
        print(score)
        scores["voter_" + '_'.join(langs)] = score
        with open(f"/home/miro/PycharmProjects/ovosauro/pretrained/accuracy.json", "w") as _f:
            json.dump(scores, _f, indent=4, sort_keys=True)

        engine.save(f"/home/miro/PycharmProjects/ovosauro/pretrained/voter_{'_'.join(langs)}.pkl")

        jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
        with AudioFile(jfk) as source:
            audio = Recognizer().record(source)

        pred = engine.recognize(audio)
        print(pred)
        print(max(pred, key=lambda k: k[1]))
        continue
        for f in os.listdir("/home/miro/dataset_dl/speech/VoxLingua107/pt_crops"):
            jfk = f"/home/miro/dataset_dl/speech/VoxLingua107/pt_crops/{f}"
            # jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
            with AudioFile(jfk) as source:
                audio = Recognizer().record(source)

            pred = engine.recognize(audio)
            print(pred)
            print(max(pred, key=lambda k: k[1]))
        # [('en', 0.19933143379085647), ('es', 0.20223241363691447),
        # ('fr', 0.36167825099149203), ('pt', 0.2367579015807368)]
