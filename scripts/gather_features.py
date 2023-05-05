import os
import pickle
import random
from threading import Thread

from ovos_utils import wait_for_exit_signal
from speech_recognition import Recognizer, AudioFile

from ovosauro import AlloSaurus


class Trainer(Thread):
    def __init__(self, sauro, feats_folder, langs, VOXLINGUA):
        super().__init__(daemon=True)
        self.feats_folder = feats_folder
        self.sauro = sauro
        self.VOXLINGUA = VOXLINGUA
        self.langs = langs

    def extract_voxlingua(self):
        for lang in self.langs:
            print(lang)
            os.makedirs(f"{self.feats_folder}/{lang}", exist_ok=True)

            path = f"{self.VOXLINGUA}/{lang}_crops"  # TODO rm _crops before publish

            for root, folders, files in os.walk(path):
                random.shuffle(files)
                for f in files:
                    if os.path.isfile(f"{feats_folder}/{lang}/{f}_feats.txt"):
                        continue

                    with AudioFile(f"{root}/{f}") as source:
                        audio = Recognizer().record(source)

                    with open(f"{feats_folder}/{lang}/{f}_feats.txt", "w") as _f:
                        _f.write(self.sauro.recognize(audio))

                    continue


                    if os.path.isfile(f"{feats_folder}/{lang}/{f}_feats.pkl"):
                        continue

                    try:
                        with open(f"{feats_folder}/{lang}/{f}_feats.pkl", "wb") as _f:
                            pickle.dump(self.sauro.extract_features(audio), _f)
                    except KeyboardInterrupt:
                        return
                    except:
                        print(f"bad file {f}")



    def run(self):
        self.extract_voxlingua()


VOXLINGUA = "/home/miro/dataset_dl/speech/VoxLingua107"
feats_folder = "/home/miro/PycharmProjects/ovosauro/scripts/feats"
sauro = AlloSaurus()

langs = [l.split("_")[0] for l in os.listdir(VOXLINGUA)]

random.shuffle(langs)
for l in langs:
    t = Trainer(sauro, feats_folder, [l], VOXLINGUA)
    t.start()


wait_for_exit_signal()
