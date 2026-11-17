Section 4.8 control experiments
First create non-destructive controlled copies and compute label-word statistics:

python scripts/prepare_section_4_8_controls.py `
  --input-dir D:\data\IEMOCAP_fold1 `
  --output-dir D:\data\IEMOCAP_fold1_controls
The audit is written to label_shortcut_audit.json. The script also creates label_word_masking and shuffled_explanation directories containing the same split filenames as the source directory.

Run each condition with the same fold, training hyperparameters, and evaluation protocol. Replace paths as needed:

python scripts/train_section_4_8.py --config src/configs/hubert_llm_racmt_contrastive_v3.py --variant label_word_masking --data-root D:\data\IEMOCAP_fold1_controls\label_word_masking --checkpoint-root D:\checkpoints\fold1
python scripts/train_section_4_8.py --config src/configs/hubert_llm_racmt_contrastive_v3.py --variant shuffled_explanation --data-root D:\data\IEMOCAP_fold1_controls\shuffled_explanation --checkpoint-root D:\checkpoints\fold1
python scripts/train_section_4_8.py --config src/configs/hubert_llm_racmt_contrastive_v3.py --variant no_explanation --data-root D:\data\IEMOCAP_fold1 --checkpoint-root D:\checkpoints\fold1
python scripts/train_section_4_8.py --config src/configs/hubert_llm_racmt_contrastive_v3.py --variant explanation_only --data-root D:\data\IEMOCAP_fold1 --checkpoint-root D:\checkpoints\fold1
For IEMOCAP, repeat these commands for all five session-independent folds and report the mean and sample standard deviation across the five test sessions.