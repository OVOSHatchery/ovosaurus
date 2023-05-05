from os.path import dirname, isfile

from ovos_utils.log import LOG

from ovos_plugin_manager.templates.transformers import AudioTransformer
from ovosaurus import OVOSaurus


class OVOSaurusLangClassifier(AudioTransformer):
    def __init__(self, config=None):
        config = config or {}
        super().__init__("ovos-audio-transformer-plugin-ovosaurus", 10, config)
        model = self.config.get("model") or "svc_tfidf_en_fi"
        if isfile(model):
            model_path = model
        else:
            # TODO - auto download from /models branch
        #    model_path = f"{dirname(dirname(__file__))}/pretrained/{model}.pkl"
        #if not isfile(model_path):
            raise ValueError(f"invalid model: {model}")
        self.engine = OVOSaurus.from_file(model_path)

    # plugin api
    def transform(self, audio_chunk):
        pred = self.engine.recognize(audio_chunk)
        best = max(pred, key=lambda k: k[1])
        LOG.info(f"Detected speech language '{best[0]}' with probability {best[1]}")
        return audio_chunk, {"stt_lang": best[0], "lang_probability": best[1]}
