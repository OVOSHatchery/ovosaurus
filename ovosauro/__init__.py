import numpy as np
from allosaurus.am.utils import move_to_tensor
from allosaurus.app import read_recognizer
from allosaurus.audio import Audio
from speech_recognition import Recognizer, AudioFile, AudioData


class AlloSaurus:
    def __init__(self):
        self.model = read_recognizer()

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


if __name__ == "__main__":
    # load your model

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    m = AlloSaurus()
    # run inference -> æ l u s ɔ ɹ s
    ph = m.recognize(audio)
    print(ph, type(ph))
    feats = m.extract_features(audio)
    print(feats)

