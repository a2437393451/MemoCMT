from configs.hubert_llm_racmt_esd import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.model_type = "MemoCMT"
        self.use_llm_explanations = False
        self.checkpoint_dir = "/root/autodl-tmp/MemoCMT/checkpoints/ESD_V1_ABLATION"
        self.name = "MemoCMT_bert_hubert_base_ESD_v1_ablation_wo_llm"
        for key, value in kwargs.items():
            setattr(self, key, value)
