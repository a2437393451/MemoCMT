from configs.base import Config as BaseConfig


class Config(BaseConfig):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 4
        self.num_epochs = 10
        self.random_seed = 0
        self.use_attention_masks = True
        self.use_dynamic_padding = True
        self.checkpoint_metric = "macro_f1"

        self.loss_type = "CrossEntropyContrastiveLoss"
        self.use_contrastive_loss = True
        self.contrastive_loss_weight = 0.1
        self.contrastive_temperature = 0.07
        self.contrastive_mode = "instance"

        self.checkpoint_dir = (
            "/root/autodl-tmp/MemoCMT/checkpoints/ESD_LLM_RACMT_CONTRASTIVE"
        )

        self.model_type = "LLMReliabilityMemoCMT"

        self.text_encoder_type = "bert"
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        self.audio_encoder_type = "hubert_base"
        self.audio_encoder_dim = 768
        self.audio_unfreeze = False

        self.llm_text_encoder_type = "bert"
        self.llm_text_encoder_dim = 768
        self.llm_text_unfreeze = False

        self.fusion_dim = 768
        self.reliability_gate_hidden_dim = 256

        self.use_llm_explanations = True
        self.llm_explanation_key = "llm_explanation"
        self.llm_explanation_max_length = 128
        self.llm_explanation_fallback_to_text = True

        self.data_name = "ESD"
        self.data_root = "/root/autodl-tmp/MemoCMT/dataset/ESD_llm_preprocessed"
        self.data_valid = "val.pkl"
        self.num_classes = 4
        self.text_max_length = 297
        self.audio_max_length = 128000

        self.name = "LLMReliabilityMemoCMT_bert_hubert_base_ESD_contrastive_v2_revised"

        for key, value in kwargs.items():
            setattr(self, key, value)
