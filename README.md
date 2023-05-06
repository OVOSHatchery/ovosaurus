# OVOSAURUS

:WIP: - just a fun idea, open during construction, might lead nowhere


classify language from spoken audio

OVOSAURUS turns an audio classification problem into a text classification problem, allowing for new models to be quickly trained on limited data for different language combinations on demand

1 - turn audio into a sequence of IPA phonemes, initial implementation uses Allosaurus, this step is lang agnostic (240 different phonemes)

2 - train a classic machine learning classifier on the phonemes (initial dataset [allosaurusVoxLingua_v0.1.csv](https://github.com/OpenVoiceOS/ovos-datasets/blob/master/text/allosaurusVoxLingua_v0.1.csv))


TODO - make it independent from Allosaurus (GPL) , abstract audio2ipa under OPM

https://github.com/OpenVoiceOS/ovos-plugin-manager/pull/147

initial plugins for testing:

- https://huggingface.co/bookbot/wav2vec2-ljspeech-gruut   (included in this repo for now)
- https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft   (included in this repo for now)
- https://github.com/xinjli/allosaurus   (included in this repo for now)

# OVOS usage

language detector for ovos dinkum listener

see [models branch for pretrained models](https://github.com/OpenVoiceOS/ovosaurus/tree/models/pretrained), download one and provide path


```javascript
"listener": {
    "audio_transformers": {
        "ovos-audio-transformer-plugin-ovosaurus": {
            "model": "https://github.com/OpenVoiceOS/ovosaurus/raw/models/pretrained/svc_de_en_es_fi_fr_pt.pkl"
        }
    }
}
```

# Training

see [models branch](https://github.com/OpenVoiceOS/ovosaurus/tree/models#training)

# Usage

not ready, open during construction, browse the code or something

```python
from speech_recognition import Recognizer, AudioFile

name = "https://github.com/OpenVoiceOS/ovosaurus/raw/models/pretrained/svc_de_en_es_fi_fr_pt.pkl"
engine = OVOSaurus.from_file(name)

# inference
jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
with AudioFile(jfk) as source:
    audio = Recognizer().record(source)

pred = engine.recognize(audio)
print(pred)  # all langs
# [('de', 0.2966126259614516), ('en', 0.32389020726666456), ('es', 0.07285172650618475), ('fi', 0.09471670467904182), ('fr', 0.08266133731016057), ('pt', 0.12926739827649664)]
print(max(pred, key=lambda k: k[1]))  # best lang
# ('en', 0.32389020726666456)
```
