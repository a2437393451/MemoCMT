from configs.hubert_llm_racmt_contrastive_v3 import Config as V3Config


class Config(V3Config):
    """URM-ER six-class IEMOCAP configuration using the EMART split."""

    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "URM_ER_IEMOCAP6_EMART"
        self.num_classes = 6
        self.data_root = "working/dataset/IEMOCAP6_EMART_llm_preprocessed"
        self.checkpoint_dir = "working/checkpoints/IEMOCAP6_EMART"
        self.fusion_head_output_type = "min"
        for key, value in kwargs.items():
            setattr(self, key, value)
