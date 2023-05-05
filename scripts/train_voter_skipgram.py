import json
import os
import random
from os.path import dirname

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from speech_recognition import Recognizer, AudioFile
from ovosauro import powerset, OVOSauro


scores = {}
try:
    with open(f"{dirname(dirname(__file__))}/pretrained/accuracy_voter.json", "r") as _f:
        scores = json.load(_f)
except:
    pass

# single clf
clf = [LogisticRegression(), KNeighborsClassifier(), MultinomialNB()]

# pre defined pipelines from ovos-classifiers
clf_pipeline = "skipgram2"
lang_combos = powerset(sorted(["en", "pt", "fr", "it", "es", "de"]))
random.shuffle(lang_combos)
dt = None

for langs in lang_combos:
    langs = list(langs)
    print(langs)

    name = f"voter_lr_kn_nb_{clf_pipeline}_{'_'.join(langs)}"
    if os.path.isfile(f"{dirname(dirname(__file__))}/pretrained/{name}.pkl"):
        continue

    engine = OVOSauro(langs, clf, clf_pipeline)

    # load dataset pre generated feats only once
    feats = f"{dirname(__file__)}/feats"
    if dt is None:
        engine.load_features(feats)
        dt = engine.lang_features
    else:
        engine.lang_features = dt

    # load data
    X, y = engine.load_data()

    # train/test split
    n = int(len(X) * 0.7)
    Xtest = X[n:]
    X = X[:n]
    ytest = y[n:]
    y = y[:n]

    # train
    engine.train(X, y)

    # test
    score = engine.classifier.score(Xtest, ytest)
    print(score)
    scores[name] = score
    with open(f"{dirname(dirname(__file__))}/pretrained/accuracy_voter.json", "w") as _f:
        json.dump(scores, _f, indent=4, sort_keys=True)

    # save
    engine.save(f"{dirname(dirname(__file__))}/pretrained/{name}.pkl")

    # inference
    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    pred = engine.recognize(audio)
    print(pred)  # all langs
    print(max(pred, key=lambda k: k[1]))  # best lang

