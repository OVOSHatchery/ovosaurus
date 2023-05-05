from os.path import dirname

from speech_recognition import Recognizer, AudioFile

from ovosauro import OVOSauro

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
