import numpy as np
from allosaurus.am.utils import move_to_tensor
from allosaurus.app import read_recognizer
from allosaurus.audio import Audio
from speech_recognition import Recognizer, AudioFile, AudioData
from ovos_plugin_manager.templates.audio2ipa import Audio2IPA
from ovos_utils.xdg_utils import xdg_data_home
from pathlib import Path

# TODO - own pluginf

class AlloSaurus(Audio2IPA):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = read_recognizer(alt_model_path=Path(f"{xdg_data_home()}/Allosaurus"))

    @staticmethod
    def read_audio(audio_data):
        """
        read_audio will read a raw wav and return an Audio object

        :param header_only: only load header without samples
        """
        assert isinstance(audio_data, AudioData)

        # initialize audio
        audio = Audio()

        # set wav header
        audio.set_header(sample_rate=audio_data.sample_rate,
                         channel_number=1,
                         sample_width=audio_data.sample_width)

        x = audio_data.get_wav_data()
        audio_bytes = np.frombuffer(x, dtype='int16')
        audio.samples = audio_bytes
        audio.sample_size = len(audio.samples)

        return audio

    def extract_features(self, audio_data):
        # load wav audio
        audio = self.read_audio(audio_data)
        feats = self.model.pm.compute(audio)
        return feats

    def recognize(self, audio_data, lang_id='ipa', topk=1, emit=1.0, timestamp=False):
        # extract features
        feat = self.extract_features(audio_data)

        # add batch dim
        feats = np.expand_dims(feat, 0)
        feat_len = np.array([feat.shape[0]], dtype=np.int32)

        tensor_batch_feat, tensor_batch_feat_len = move_to_tensor([feats, feat_len], self.model.config.device_id)

        tensor_batch_lprobs = self.model.am(tensor_batch_feat, tensor_batch_feat_len)

        if self.model.config.device_id >= 0:
            batch_lprobs = tensor_batch_lprobs.cpu().detach().numpy()
        else:
            batch_lprobs = tensor_batch_lprobs.detach().numpy()

        token = self.model.lm.compute(batch_lprobs[0], lang_id, topk, emit=emit, timestamp=timestamp)
        return token

    def get_ipa(self, audio_data):
        return self.recognize(audio_data)


if __name__ == "__main__":
    # load your model

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    m = AlloSaurus()
    # run inference -> æ l u s ɔ ɹ s
    ph = m.get_ipa(audio)
    print(ph)
