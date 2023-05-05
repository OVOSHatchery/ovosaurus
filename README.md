# OVOSAURUS

:WIP: - just a fun idea, open during construction, might lead nowhere


datasets:


# Training

see scripts/gather_features.py for the code used to generate the dataset, a subset of VoxLingua107 was used

see train_svc.py for initial implementation using tfidf + SVC

see train_voter.py for initial implementation using a soft voting classifier (SVC, DecisionTree, LogisticRegression)

# Usage

not ready, open during construction, browse the code or something

```python
from os.path import dirname

from speech_recognition import Recognizer, AudioFile

from ovosaurus import OVOSauro

name = "svc_tfidf_en_fi"
engine = OVOSauro.from_file(f"{dirname(dirname(__file__))}/pretrained/{name}.pkl")

# inference
jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
with AudioFile(jfk) as source:
    audio = Recognizer().record(source)

pred = engine.recognize(audio)
print(pred)  # all langs
# [('en', 0.559955228299166), ('fi', 0.44004477170083406)]
print(max(pred, key=lambda k: k[1]))  # best lang
# ('en', 0.559955228299166)
```