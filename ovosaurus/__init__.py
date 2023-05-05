import os
import random
from itertools import chain, combinations

from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier, SklearnOVOSVotingClassifier
from ovos_plugin_manager.audio2ipa import OVOSAudio2IPAFactory
from sklearn.svm import SVC
from speech_recognition import AudioData


# TODO - move to ovos-utils
def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return [_ for _ in chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)) if len(_) > 1]


class OVOSaurus:
    def __init__(self, langs, clf=None, pipeline="tfidf"):
        self.langs = langs
        clf = clf or SVC(probability=True)

        if isinstance(clf, list):
            self.classifier = SklearnOVOSVotingClassifier(voter_clfs=clf, pipeline_id=pipeline)
        else:
            self.classifier = SklearnOVOSClassifier(pipeline_id=pipeline, pipeline_clf=clf)

        self.lang_features = {}

        # TODO - plugins
        try:
            self.sauro = OVOSAudio2IPAFactory.create()
        except:
            from ovosaurus.ovos_audio2ipa_plugin_allosaurus import AlloSaurus
            self.sauro = AlloSaurus()

    def recognize(self, audio_data, unknown=False):
        audio_data = AudioData(audio_data, 16000, 2)
        feats = [self.sauro.recognize(audio_data)]
        classes = self.classifier.clf.classes_
        probs = self.classifier.clf.predict_proba(feats)[0]
        a = list(zip(classes, probs))
        if not unknown:
            a = [_ for _ in a if _[0] != "other"]
            total = sum(_[1] or 0 for _ in a)
            b = 1 / (len(classes) - 1)
            a = [(_[0], _[1] / total) if total else (_[0], b) for _ in a]
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
        self.langs = self.classifier.clf.classes_

    @staticmethod
    def from_file(path):
        clf = OVOSaurus([])
        clf.load(path)
        return clf
