from configs.hubert_llm_racmt_contrastive_esd import Config as V2ESDConfig


class Config(V2ESDConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_v3_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_v3_args(self):
        self.name = (
            "LLMReliabilityMemoCMT_bert_hubert_base_ESD_contrastive_v3_uncertainty"
        )
        self.checkpoint_dir = (
            "/workspace/MemoCMTdataset/MemoCMT/checkpoints/ESD_LLM_RACMT_V3"
        )

        self.contrastive_mode = "supervised"
        self.contrastive_loss_weight = 0.03
        self.contrastive_temperature = 0.1

        self.use_uncertainty_weighting = True
        self.use_uncertainty_fusion_weighting = True
        self.use_uncertainty_contrastive_weighting = True
        self.uncertainty_hidden_dim = 192
        self.uncertainty_loss_weight = 0.001
