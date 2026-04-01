# GeneCompass 评估脚本使用说明

这个脚本用于评估已训练的 GeneCompass 模型在 arrow 数据集上的损失。

## 使用方法

### 基本用法

```bash
python evaluate_genecompass.py \
    --token_dict_path "../prior_knowledge/h&m_token1000W.pickle" \
    --dataset_directory "../scdata/output/patch1" \
    --model_checkpoint_path "../model_weight/base" \
    --eval_micro_batch_size_per_gpu 10 \
    --output_directory "./eval_output"
```

### 参数说明

- `--token_dict_path`: token字典文件路径（必需）
- `--dataset_directory`: 数据集目录路径，包含arrow文件和sorted_length.pickle（必需）
- `--model_checkpoint_path`: 模型检查点目录路径，包含pytorch_model.bin（必需）
- `--eval_micro_batch_size_per_gpu`: 每个GPU的评估批次大小（默认：10）
- `--output_directory`: 评估结果输出目录（默认：./eval_output）
- `--run_name`: 运行名称，用于日志记录（默认：evaluation）
- `--fp16`: 是否使用混合精度（可选）
- `--dataloader_num_workers`: 数据加载器工作进程数（默认：0）

### 示例

使用你现有的arrow文件进行评估：

```bash
cd /media/ubuntu/sda/GeneCompass-main/pretraining

python evaluate_genecompass.py \
    --token_dict_path "../prior_knowledge/h&m_token1000W.pickle" \
    --dataset_directory "../scdata/output/patch1" \
    --model_checkpoint_path "../model_weight/base" \
    --eval_micro_batch_size_per_gpu 10 \
    --output_directory "./eval_output" \
    --run_name "evaluation_patch1" \
    --fp16
```

## 输出

脚本会输出以下评估指标：
- `eval_loss`: 总体损失
- `eval_id_loss`: ID损失
- `eval_value_loss`: 值损失
- 其他相关指标

所有结果会打印到控制台，并保存在输出目录中。

## 注意事项

1. 确保数据集目录包含 `sorted_length.pickle` 文件。如果不存在，脚本会自动生成。
2. 确保模型检查点目录包含 `pytorch_model.bin` 和 `config.json`。
3. 如果使用GPU，确保CUDA可用。
4. 如果遇到内存问题，可以减小 `--eval_micro_batch_size_per_gpu` 的值。

