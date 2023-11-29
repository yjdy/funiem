set -e
date=$(date +%Y%m%d-%H:%M)

accelerate launch --config_file ./config/accelerate_config.yaml finetune_sentence_transformers.py