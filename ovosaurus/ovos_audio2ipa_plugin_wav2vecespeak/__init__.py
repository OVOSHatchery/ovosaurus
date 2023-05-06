from itertools import groupby

import numpy as np
import torch
from speech_recognition import Recognizer, AudioFile, AudioData
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor

from ovos_plugin_manager.templates.audio2ipa import Audio2IPA


# TODO - own plugin
class Wav2VecEspeak(Audio2IPA):
    def __init__(self, config=None):
        super().__init__(config)
        checkpoint = "facebook/wav2vec2-lv-60-espeak-cv-ft"
        self.model = AutoModelForCTC.from_pretrained(checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    @staticmethod
    def decode_phonemes(
            ids: torch.Tensor, processor: Wav2Vec2Processor, ignore_stress: bool = False
    ) -> str:
        """CTC-like decoding. First removes consecutive duplicates, then removes special tokens."""
        # removes consecutive duplicates
        ids = [id_ for id_, _ in groupby(ids)]

        special_token_ids = processor.tokenizer.all_special_ids + [
            processor.tokenizer.word_delimiter_token_id
        ]
        # converts id to token, skipping special tokens
        phonemes = [processor.decode(id_) for id_ in ids if id_ not in special_token_ids]

        # joins phonemes
        prediction = " ".join(phonemes)

        # whether to ignore IPA stress marks
        if ignore_stress == True:
            prediction = prediction.replace("ˈ", "").replace("ˌ", "")

        return prediction

    @staticmethod
    def audiochunk2array(audio_data):
        # Convert buffer to float32 using NumPy
        audio_as_np_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

        # Normalise float32 array so that values are between -1.0 and +1.0
        max_int16 = 2 ** 15
        data = audio_as_np_float32 / max_int16
        return data

    def recognize(self, audio_data):
        if isinstance(audio_data, AudioData):
            audio_data = audio_data.frame_data

        audio_array = self.audiochunk2array(audio_data)

        inputs = self.processor(audio_array, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs["input_values"]).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return self.decode_phonemes(predicted_ids[0], self.processor, ignore_stress=True)
        # => should give 'b ɪ k ʌ z j u ɚ z s l i p ɪ ŋ ɪ n s t ɛ d ə v k ɔ ŋ k ɚ ɪ ŋ ð ə l ʌ v l i ɹ z p ɹ ɪ n s ə s h æ z b ɪ k ʌ m ə v f ɪ t ə l w ɪ θ n b oʊ p ɹ ə ʃ æ ɡ i s ɪ t s ð ɛ ɹ ə k u ɪ ŋ d ʌ v'

    def get_ipa(self, audio_data):
        return self.recognize(audio_data)


if __name__ == "__main__":
    # load your model

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    m = Wav2VecEspeak()
    # run inference -> æ l u s ɔ ɹ s
    ph = m.get_ipa(audio)
    print(ph)
