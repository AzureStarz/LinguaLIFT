# Environment
conda create -n lingualift python=3.10 -y
cd LLaMA-Factory && pip install -e ./
cd lm-evaluation-harness && pip install -e ./
cd model && ln -s ./*.py LLaMA-Factory/src/llamafactory/model

# Directory
LLaMA-Factory/ training framework
lm-evaluation-harness evaluation framework
model/ **the specific implementation of LinguaLIFT**
python_scripts/ contains evaluation python scripts
scripts/ contains linux shell scripts

# Data
https://drive.google.com/file/d/1Z3Qt9wopjW2cLDX3UADYyCLgOw4zt8zK/view?usp=sharing
Download the data file and update the LLaMA-Factory/data/dataset_info.jsondataset_info.json

# Train
update the reference files path and run
```shell
sh scripts/train/train_math.sh
```

# Evaluation
update the reference fiels path and run
```shell
sh scripts/eval/mgsm_eval.sh
```