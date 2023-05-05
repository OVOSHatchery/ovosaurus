from ovosauro.clf import *



scores = {}
try:
    with open(f"/home/miro/PycharmProjects/ovosauro/pretrained/accuracy.json", "r") as _f:
        scores = json.load(_f)
except:
    pass

# single clf
clf = SVC(probability=True)

# pre defined pipelines from ovos-classifiers
clf_pipeline = "tfidf"  # "tfidf"
lang_combos = powerset(sorted(["en", "pt", "fr", "it", "es", "de", "fi", "nl"]))
random.shuffle(lang_combos)
dt = None

for langs in lang_combos:
    langs = list(langs)
    print(langs)
    if os.path.isfile(f"/home/miro/PycharmProjects/ovosauro/pretrained/svc_{'_'.join(langs)}.pkl"):
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
    scores["svc_" + '_'.join(langs)] = score
    with open(f"/home/miro/PycharmProjects/ovosauro/pretrained/accuracy.json", "w") as _f:
        json.dump(scores, _f, indent=4, sort_keys=True)

    engine.save(f"/home/miro/PycharmProjects/ovosauro/pretrained/svc_{'_'.join(langs)}.pkl")

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
