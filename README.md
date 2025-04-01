# 🚀 LinguaLIFT: Fine-Tuning LLaMA for Multilingual Mastery

## 📚 Environment Setup
```bash
# Create a conda environment
conda create -n lingualift python=3.10 -y

# Install LLaMA-Factory dependencies
cd LLaMA-Factory && pip install -e ./

# Install lm-evaluation-harness dependencies
cd lm-evaluation-harness && pip install -e ./

# Link model files to LLaMA-Factory
cd model && ln -s ./*.py LLaMA-Factory/src/llamafactory/model
```

---

## 📂 Directory Structure

```
📁 LLaMA-Factory/          # Training framework
📁 lm-evaluation-harness/  # Evaluation framework
📁 model/                  # Specific implementation of LinguaLIFT
📁 python_scripts/         # Python scripts for evaluation
📁 scripts/                # Shell scripts for automation
```

---

## 📥 Data Preparation
1. Download the dataset.
https://drive.google.com/file/d/1Z3Qt9wopjW2cLDX3UADYyCLgOw4zt8zK/view?usp=sharing
2. Update the dataset information in:
```
LLaMA-Factory/data/dataset_info.json
```

---

## 🎯 Training
1. Update reference file paths as needed.
2. Run the training script:
```bash
sh scripts/train/train_math.sh
```

---

## 🧪 Evaluation
1. Update reference file paths.
2. Run the evaluation script:
```bash
sh scripts/eval/mgsm_eval.sh
```

---

## 💡 Notes
- Ensure all dependencies are installed and file paths are correctly configured before running scripts.
- For additional customization or troubleshooting, refer to the respective framework documentation.
