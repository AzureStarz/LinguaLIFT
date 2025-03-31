#!/bin/env bash
ckpt_path=${1:-"path/to/ckpt"}
output_path=${2:-"eval_outputs/mgsm"}
enc_tokenizer_path=${3:-"path/to/mt5_tokenizer"}
template_name=${4:-"metamath"}

mkdir -p ${output_path}

python python_scripts/eval_langbridge.py \
  --checkpoint_path ${ckpt_path} \
  --enc_tokenizer ${enc_tokenizer_path} \
  --tasks mgsm_en,mgsm_es,mgsm_fr,mgsm_de,mgsm_ru,mgsm_zh,mgsm_ja,mgsm_th,mgsm_sw,mgsm_bn,mgsm_te \
  --instruction_template ${template_name} \
  --batch_size 1 \
  --output_path ${output_path} \
  --device cuda:0 \
  --no_cache