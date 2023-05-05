# OVOSAURUS

:WIP: - just a fun idea, open during construction, might lead nowhere


classify language from spoken audio

OVOSAURUS turns an audio classification problem into a text classification problem, allowing for new models to be quickly trained on limited data for different language combinations on demand

1 - turn audio into a sequence of IPA phonemes, initial implementation uses Allosaurus, this step is lang agnostic (240 different phonemes)

2 - train a classic machine learning classifier on the phonemes (initial dataset [allosaurusVoxLingua_v0.1.csv](https://github.com/OpenVoiceOS/ovos-datasets/blob/master/text/allosaurusVoxLingua_v0.1.csv))


TODO - make it independent from Allosaurus (GPL) , abstract audio2ipa under OPM

https://github.com/OpenVoiceOS/ovos-plugin-manager/pull/147

initial plugins for testing:

- https://huggingface.co/bookbot/wav2vec2-ljspeech-gruut
- https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
- https://github.com/xinjli/allosaurus/   (included here for now)

# OVOS usage

language detector for ovos dinkum listener

see [models branch for pretrained models](https://github.com/OpenVoiceOS/ovosaurus/tree/models/pretrained), download one and provide path

TODO - auto download

```javascript
"listener": {
    "audio_transformers": {
        "ovos-audio-transformer-plugin-ovosaurus": {
            "model": "path/to/svc_tfidf_en_fr_pt.pkl"
        }
    }
}
```

# Training

see [models branch](https://github.com/OpenVoiceOS/ovosaurus/tree/models#training)

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
