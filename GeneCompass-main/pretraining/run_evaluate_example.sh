#!/bin/bash

# 评估脚本使用示例
# 使用你的arrow数据集评估GeneCompass模型

python evaluate_genecompass.py \
    --token_dict_path "../prior_knowledge/h&m_token1000W.pickle" \
    --dataset_directory "../scdata/output/patch1" \
    --model_checkpoint_path "../model_weight/base" \
    --eval_micro_batch_size_per_gpu 10 \
    --output_directory "./eval_output" \
    --run_name "evaluation_patch1" \
    --fp16

