#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
import random
import sys
import tempfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    disable_caching,
    load_from_disk,
)
from transformers import BertConfig, TrainingArguments

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

os.environ.setdefault("WANDB_MODE", "offline")
disable_caching()

from genecompass import GenecompassPretrainer, BertForMaskedLM  # noqa: E402
from genecompass.utils import load_prior_embedding  # noqa: E402


def prepare_dataset_from_csv(
    csv_path: str,
    token_dictionary: dict,
    pad_token_id: int,
    max_sequence_length: int,
    species_label: int,
) -> Tuple[Dataset, List[int]]:
    # 先读取header确定列数
    max_cells = 1000
    header = pd.read_csv(csv_path, nrows=0)
    total_cells_in_csv = header.shape[1] - 1  # 减去第一列基因列
    actual_num_cells = min(max_cells, total_cells_in_csv)
    
    if actual_num_cells < total_cells_in_csv:
        print(f"CSV 共有 {total_cells_in_csv} 个细胞，仅读取前 {actual_num_cells} 个细胞列")
        # 只读取第一列（基因ID）+ 前1000个细胞列
        usecols = [0] + list(range(1, 1 + actual_num_cells))
        df = pd.read_csv(csv_path, usecols=usecols)
    else:
        print(f"CSV 共有 {total_cells_in_csv} 个细胞，读取全部细胞")
        df = pd.read_csv(csv_path)
    
    if df.empty:
        raise ValueError(f"CSV 文件 {csv_path} 为空，无法评估。")

    gene_ids_raw = df.iloc[:, 0].astype(str).tolist()
    matrix = df.iloc[:, 1:].to_numpy(dtype=np.float32)

    token_ids: List[int] = []
    valid_rows: List[int] = []
    for idx, gene_id in enumerate(gene_ids_raw):
        gene_key = gene_id.split("|")[0]
        token_id = token_dictionary.get(gene_key)
        if token_id is None:
            continue
        token_ids.append(token_id)
        valid_rows.append(idx)

    if len(valid_rows) == 0:
        raise ValueError("CSV 中的基因在 token 字典里均未找到映射。")

    matrix = matrix[valid_rows, :]
    token_ids = np.asarray(token_ids, dtype=np.int32)
    num_cells = matrix.shape[1]

    input_ids = np.full(
        (num_cells, max_sequence_length), pad_token_id, dtype=np.int32
    )
    values = np.zeros((num_cells, max_sequence_length), dtype=np.float32)
    lengths: List[int] = []
    species = np.full((num_cells, 1), species_label, dtype=np.int16)

    print(f"开始转换 {num_cells} 个细胞为模型输入...")
    step = max(1, num_cells // 10)
    for cell_idx in range(num_cells):
        cell_expr = matrix[:, cell_idx]
        nonzero = np.nonzero(cell_expr)[0]
        if len(nonzero) == 0:
            lengths.append(0)
            continue
        sorted_indices = nonzero[np.argsort(-cell_expr[nonzero])]
        sorted_tokens = token_ids[sorted_indices]
        sorted_values = cell_expr[sorted_indices]
        seq_len = min(len(sorted_tokens), max_sequence_length)
        input_ids[cell_idx, :seq_len] = sorted_tokens[:seq_len]
        values[cell_idx, :seq_len] = sorted_values[:seq_len]
        lengths.append(int(seq_len))
        if (cell_idx + 1) % step == 0 or cell_idx + 1 == num_cells:
            print(f"  已处理 {cell_idx + 1}/{num_cells} 个细胞")

    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "values": Sequence(Value("float32")),
            "length": Sequence(Value("int16")),
            "species": Sequence(Value("int16")),
        }
    )

    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids.tolist(),
            "values": values.tolist(),
            "length": [[l] for l in lengths],
            "species": species.tolist(),
        },
        features=features,
    )

    sorted_lengths = sorted(lengths)
    return dataset, sorted_lengths


def main(args):
    random.seed(args.seed_num)
    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_val)
    torch.cuda.manual_seed_all(args.seed_val)

    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.world_rank = int(os.environ.get("RANK", "0"))
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))

    training_output_dir = os.path.join(args.output_directory, "models", args.run_name)
    logging_dir = os.path.join(args.output_directory, "runs", args.run_name)
    model_output_dir = os.path.join(training_output_dir, "models")

    if args.world_rank == 0 and (args.do_train or args.save_model):
        os.makedirs(training_output_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)

    with open(args.token_dict_path, "rb") as fp:
        token_dictionary = pickle.load(fp)

    knowledges = dict()
    out = load_prior_embedding(token_dictionary_or_path=args.token_dict_path)
    knowledges["promoter"] = out[0]
    knowledges["co_exp"] = out[1]
    knowledges["gene_family"] = out[2]
    knowledges["peca_grn"] = out[3]
    knowledges["homologous_gene_human2mouse"] = out[4]

    pad_token_id = token_dictionary.get("<pad>")
    config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "attention_probs_dropout_prob": 0.02,
        "hidden_dropout_prob": 0.02,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "max_position_embeddings": args.max_sequence_length,
        "model_type": "bert",
        "num_attention_heads": 12,
        "pad_token_id": pad_token_id,
        "vocab_size": len(token_dictionary),
        "use_values": True,
        "use_promoter": True,
        "use_co_exp": True,
        "use_gene_family": True,
        "use_peca_grn": True,
        "warmup_steps": args.warmup_steps,
        "emb_warmup_steps": args.emb_warmup_steps,
        "use_cls_token": True,
    }

    model_config = BertConfig(**config)
    if args.pretrained_model_path:
        model = BertForMaskedLM.from_pretrained(
            args.pretrained_model_path,
            knowledges=knowledges,
            config=model_config,
        )
    else:
        model = BertForMaskedLM(model_config, knowledges=knowledges)

    if args.do_train:
        model.train()
    else:
        model.eval()

    training_args = {
        "run_name": args.run_name,
        "fp16": args.fp16,
        "fp16_opt_level": "O1",
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataloader_num_workers": args.dataloader_num_workers,
        "learning_rate": args.max_learning_rate,
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "group_by_length": args.group_by_length,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.train_micro_batch_size_per_gpu,
        "per_device_eval_batch_size": args.eval_micro_batch_size_per_gpu,
        "num_train_epochs": args.num_train_epochs,
        "save_strategy": "steps" if args.save_model else "no",
        "save_steps": args.save_steps if args.save_model else None,
        "logging_steps": 100,
        "output_dir": training_output_dir,
        "logging_dir": logging_dir,
    }
    training_args = TrainingArguments(**training_args)

    temp_dir = None
    if args.csv_path:
        dataset, sorted_lengths = prepare_dataset_from_csv(
            csv_path=args.csv_path,
            token_dictionary=token_dictionary,
            pad_token_id=pad_token_id,
            max_sequence_length=args.max_sequence_length,
            species_label=args.species_label,
        )
        temp_dir = tempfile.mkdtemp(prefix="genecompass_eval_")
        example_lengths_file = os.path.join(temp_dir, "sorted_length.pickle")
        with open(example_lengths_file, "wb") as f:
            pickle.dump(sorted_lengths, f)
        train_dataset = dataset
    else:
        if not args.dataset_directory:
            raise ValueError("请提供 --csv_path 或 --dataset_directory。")
        train_dataset = load_from_disk(args.dataset_directory)
        example_lengths_file = os.path.join(
            args.dataset_directory, "sorted_length.pickle"
        )

    if args.world_rank == 0:
        if args.do_train:
            print("开始训练。")
        elif args.do_eval:
            print("开始评估。")

    trainer = GenecompassPretrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        example_lengths_file=example_lengths_file,
        token_dictionary=token_dictionary,
    )

    if args.do_train:
        trainer.train()

    if args.do_eval:
        total = len(train_dataset)
        step = max(1, total // 10)
        print(f"评估前检查：数据集中共 {total} 个样本。")
        for idx in range(0, total, step):
            print(f"  评估准备进度：{min(idx + step, total)}/{total}")
        metrics = trainer.evaluate()
        if args.world_rank == 0:
            print("Evaluation metrics:", metrics)

    if args.save_model:
        trainer.save_model(model_output_dir)

    if temp_dir and os.path.isdir(temp_dir):
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="genecompass_eval", type=str)
    parser.add_argument("--seed_num", type=int, default=0)
    parser.add_argument("--seed_val", type=int, default=42)
    parser.add_argument("--token_dict_path", type=str, required=True)
    parser.add_argument("--dataset_directory", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--eval_micro_batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--max_learning_rate", type=float, default=5e-5)
    parser.add_argument("--min_learning_rate", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--emb_warmup_steps", type=int, default=10000)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--eval_steps", type=int, default=100000)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100000)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--max_sequence_length", type=int, default=2048)
    parser.add_argument("--species_label", type=int, default=1)
    parser.add_argument("--group_by_length", action="store_true", default=False)
    parser.add_argument("--pretrained_model_path", type=str, default=None)

    args = parser.parse_args()

    if not args.do_train and not args.do_eval:
        args.do_eval = True

    main(args)

