from configs.hubert_llm_racmt_esd import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.use_reliability_gate = False
        self.checkpoint_dir = "/workspace/MemoCMTdataset/MemoCMT/checkpoints/ESD_V1_ABLATION"
        self.name = "LLMReliabilityMemoCMT_bert_hubert_base_ESD_v1_ablation_wo_gate"
        for key, value in kwargs.items():
            setattr(self, key, value)
