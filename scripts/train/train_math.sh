#!/bin/bash

project_dir=/path/to/LinguaLIFT/LLaMA-Factory
cd ${project_dir}

save_dir=${project_dir}/saves

# # 1st stage sft
config_file=${project_dir}/examples/1st-stage.yaml
llamafactory-cli train ${config_file}

# 2nd stage sft
config_file=${project_dir}/examples/2nd-stage.yaml
llamafactory-cli train ${config_file}
