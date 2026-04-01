#!/usr/bin/env python
# coding: utf-8

import os
import sys
# 硬编码项目根路径，根据实际情况修改
project_root = "/mnt/solid1/GeneCompass-main/"
sys.path.insert(0, project_root)

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"
os.environ["WANDB_MODE"]="offline"
import pickle
import random
import argparse

import torch
import numpy as np

from transformers import BertConfig, TrainingArguments
from datasets import load_from_disk, disable_caching
disable_caching()
from genecompass import GenecompassPretrainer, BertForMaskedLM
from genecompass.utils import load_prior_embedding


def main(args):
    # Set seeds
    random.seed(args.seed_num)
    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed_val)

    # Load gene_id and prior knowledge:token dictionary 
    with open(args.token_dict_path, "rb") as fp:
        token_dictionary = pickle.load(fp)

    knowledges = dict()
    out = load_prior_embedding(token_dictionary_or_path=args.token_dict_path)
    knowledges['promoter'] = out[0]
    knowledges['co_exp'] = out[1]
    knowledges['gene_family'] = out[2]
    knowledges['peca_grn'] = out[3]
    knowledges['homologous_gene_human2mouse'] = out[4]

    # Model configuration
    config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "attention_probs_dropout_prob": 0.02,
        "hidden_dropout_prob": 0.02,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "max_position_embeddings": 2048,
        "model_type": "bert",
        "num_attention_heads": 12,
        "pad_token_id": token_dictionary.get("<pad>"),
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
    
    # Load pretrained model
    print(f"Loading pretrained model from {args.model_checkpoint_path}")
    model = BertForMaskedLM.from_pretrained(
        args.model_checkpoint_path,
        knowledges=knowledges,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    # Set to eval mode
    model = model.eval()

    # Define the training arguments (for evaluation only)
    training_args = {
        "run_name": args.run_name,
        "fp16": args.fp16,
        "fp16_opt_level": "O1",
        "dataloader_num_workers": args.dataloader_num_workers,
        "do_eval": True,
        "do_train": False,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "per_device_eval_batch_size": args.eval_micro_batch_size_per_gpu,
        "output_dir": args.output_directory,
    }
    training_args = TrainingArguments(**training_args)

    # Load evaluation dataset
    print(f"Loading dataset from {args.dataset_directory}")
    eval_dataset = load_from_disk(args.dataset_directory)
    example_lengths_file = os.path.join(args.dataset_directory, 'sorted_length.pickle')
    
    if not os.path.exists(example_lengths_file):
        print(f"Warning: sorted_length.pickle not found at {example_lengths_file}")
        print("Creating length file...")
        # Calculate lengths for each example (keep original order, not sorted)
        # Check if 'length' field exists in dataset
        if 'length' in eval_dataset[0]:
            lengths = [example['length'] for example in eval_dataset]
        else:
            # Calculate from input_ids
            lengths = []
            for i in range(len(eval_dataset)):
                example = eval_dataset[i]
                if 'input_ids' in example:
                    lengths.append(len(example['input_ids']))
                else:
                    raise ValueError(f"Example {i} does not have 'input_ids' field")
        with open(example_lengths_file, "wb") as f:
            pickle.dump(lengths, f)
        print(f"Saved lengths to {example_lengths_file} (total: {len(lengths)} examples)")

    print(f"Dataset loaded. Number of examples: {len(eval_dataset)}")

    # Define the Huggingface trainer
    # For evaluation, we can use eval_dataset as train_dataset (trainer won't actually train)
    trainer = GenecompassPretrainer(
        model=model,
        args=training_args,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        example_lengths_file=example_lengths_file,
        token_dictionary=token_dictionary,
    )

    # Start evaluation
    print("Starting evaluation...")
    eval_results = trainer.evaluate()
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for key, value in eval_results.items():
        print(f"{key}: {value}")
    print("="*50)
    
    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeneCompass model on arrow dataset")
    parser.add_argument("--run_name", default="evaluation", type=str, help="Run name for logging")
    parser.add_argument('--seed_num', type=int, default=0, help="Random seed for numpy")
    parser.add_argument('--seed_val', type=int, default=42, help="Random seed for torch")
    parser.add_argument("--token_dict_path", default=None, type=str, required=True,
                        help="Path to token dictionary pickle file")
    parser.add_argument("--dataset_directory", default=None, type=str, required=True,
                        help="Path to dataset directory containing arrow file")
    parser.add_argument("--model_checkpoint_path", default=None, type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--eval_micro_batch_size_per_gpu", default=10, type=int,
                        help="Evaluation batch size per GPU")
    parser.add_argument("--dataloader_num_workers", default=0, type=int,
                        help="Number of dataloader workers")
    parser.add_argument("--output_directory", default="./eval_output", type=str,
                        help="Output directory for evaluation results")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Warmup steps (for config, not used in eval)")
    parser.add_argument("--emb_warmup_steps", default=10000, type=int,
                        help="Embedding warmup steps (for config, not used in eval)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")

    args = parser.parse_args()

    main(args)
