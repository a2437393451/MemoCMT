from configs.hubert_llm_racmt import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.use_reliability_gate = False
        self.checkpoint_dir = "/root/autodl-tmp/MemoCMT/checkpoints/IEMOCAP_V1_ABLATION"
        self.name = "LLMReliabilityMemoCMT_bert_hubert_base_v1_ablation_wo_gate"
        for key, value in kwargs.items():
            setattr(self, key, value)
