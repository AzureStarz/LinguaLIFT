#!/bin/env bash
ckpt_path=${1:-"path/to/ckpt"}
output_path=${2:-"eval_outputs/msvamp"}
enc_tokenizer_path=${3:-"path/to/mt5_tokenizer"}
template_name=${4:-"metamath"}

mkdir -p ${output_path}

python python_scripts/eval_langbridge.py \
  --checkpoint_path ${ckpt_path} \
  --enc_tokenizer ${enc_tokenizer_path} \
  --tasks msvamp_en,msvamp_es,msvamp_fr,msvamp_de,msvamp_ru,msvamp_zh,msvamp_ja,msvamp_th,msvamp_sw,msvamp_bn \
  --instruction_template ${template_name} \
  --batch_size 1 \
  --output_path ${output_path} \
  --device cuda:0 \
  --no_cache