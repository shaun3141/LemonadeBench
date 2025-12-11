#!/usr/bin/env python3
# Copyright (c) 2025 LemonadeBench Contributors
# BSD-3-Clause License

"""
LoRA fine-tuning script for LemonadeBench.

Uses Unsloth for efficient training of large language models
on trajectory data from the Lemonade Stand environment.

Usage:
    # Train with default config
    python -m lemonade_bench.rft.train

    # Train with custom config
    python -m lemonade_bench.rft.train --config path/to/config.yaml

    # Collect data and train
    python -m lemonade_bench.rft.train --collect-data --provider anthropic

Requirements:
    pip install unsloth transformers datasets trl peft
    
    Or install with the rft extras:
    pip install -e ".[rft]"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from .config import RFTConfig, DEFAULT_CONFIG
from .formatting import LEMONADE_SYSTEM_PROMPT

console = Console()


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    
    try:
        import trl
    except ImportError:
        missing.append("trl")
    
    try:
        import peft
    except ImportError:
        missing.append("peft")
    
    # Unsloth is optional but recommended
    has_unsloth = True
    try:
        import unsloth
    except ImportError:
        has_unsloth = False
    
    if missing:
        console.print(f"[red]Missing required dependencies: {', '.join(missing)}[/red]")
        console.print("[yellow]Install with: pip install -e '.[rft]'[/yellow]")
        sys.exit(1)
    
    return has_unsloth


def load_dataset_from_jsonl(train_path: str, val_path: str | None = None):
    """Load dataset from JSONL files."""
    from datasets import Dataset, DatasetDict
    
    def load_jsonl(path: str) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
        return samples
    
    train_data = load_jsonl(train_path)
    
    # Convert to HF Dataset format
    # Extract messages for chat template
    train_formatted = []
    for sample in train_data:
        train_formatted.append({
            "messages": sample["messages"],
            "id": sample.get("id", ""),
        })
    
    train_dataset = Dataset.from_list(train_formatted)
    
    if val_path and Path(val_path).exists():
        val_data = load_jsonl(val_path)
        val_formatted = [{"messages": s["messages"], "id": s.get("id", "")} for s in val_data]
        val_dataset = Dataset.from_list(val_formatted)
        return DatasetDict({"train": train_dataset, "validation": val_dataset})
    
    return DatasetDict({"train": train_dataset})


def train_with_unsloth(config: RFTConfig, train_path: str, val_path: str | None = None):
    """Train using Unsloth for maximum efficiency."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    console.print(f"[cyan]Loading model: {config.model.model_name}[/cyan]")
    
    # Determine dtype
    import torch
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.model.dtype)
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_name,
        max_seq_length=config.model.max_seq_length,
        dtype=dtype,
        load_in_4bit=config.model.load_in_4bit,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    console.print("[green]Model loaded successfully[/green]")
    
    # Apply LoRA
    console.print(f"[cyan]Applying LoRA with rank={config.lora.r}[/cyan]")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        use_rslora=config.lora.use_rslora,
        use_gradient_checkpointing=config.lora.gradient_checkpointing_method,
    )
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(
        f"[green]Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)[/green]"
    )
    
    # Set up chat template for Qwen
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",  # Qwen3 uses same template
    )
    
    # Load dataset
    console.print(f"[cyan]Loading dataset from {train_path}[/cyan]")
    dataset = load_dataset_from_jsonl(train_path, val_path)
    
    # Format dataset for training
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    
    dataset = dataset.map(format_chat)
    
    console.print(f"[green]Dataset loaded: {len(dataset['train'])} training samples[/green]")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        optim=config.training.optim,
        seed=config.training.seed,
        bf16=config.model.dtype == "bfloat16",
        fp16=config.model.dtype == "float16",
        report_to="none",  # Disable wandb etc unless explicitly configured
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        args=training_args,
        max_seq_length=config.model.max_seq_length,
        dataset_text_field="text",
        packing=False,  # Disable packing for chat data
    )
    
    # Train
    console.print("[cyan]Starting training...[/cyan]")
    trainer.train()
    
    # Save
    console.print(f"[cyan]Saving model to {config.training.output_dir}[/cyan]")
    model.save_pretrained(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    
    # Optionally save merged model
    merged_dir = Path(config.training.output_dir) / "merged"
    console.print(f"[cyan]Saving merged model to {merged_dir}[/cyan]")
    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    
    console.print("[green]Training complete![/green]")
    
    return model, tokenizer


def train_with_peft(config: RFTConfig, train_path: str, val_path: str | None = None):
    """Train using standard PEFT/transformers (fallback if Unsloth not available)."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    
    console.print(f"[cyan]Loading model: {config.model.model_name}[/cyan]")
    console.print("[yellow]Using standard PEFT (install Unsloth for faster training)[/yellow]")
    
    # Quantization config
    bnb_config = None
    if config.model.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.model.dtype == "bfloat16" else torch.float16,
    )
    
    if config.model.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    console.print("[green]Model loaded successfully[/green]")
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load dataset
    console.print(f"[cyan]Loading dataset from {train_path}[/cyan]")
    dataset = load_dataset_from_jsonl(train_path, val_path)
    
    # Format dataset
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    
    dataset = dataset.map(format_chat)
    
    console.print(f"[green]Dataset loaded: {len(dataset['train'])} training samples[/green]")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        optim=config.training.optim,
        seed=config.training.seed,
        bf16=config.model.dtype == "bfloat16",
        fp16=config.model.dtype == "float16",
        gradient_checkpointing=config.lora.use_gradient_checkpointing,
        report_to="none",
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        args=training_args,
        max_seq_length=config.model.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    # Train
    console.print("[cyan]Starting training...[/cyan]")
    trainer.train()
    
    # Save
    console.print(f"[cyan]Saving model to {config.training.output_dir}[/cyan]")
    model.save_pretrained(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    
    console.print("[green]Training complete![/green]")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for LemonadeBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to training data JSONL file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name/path to fine-tune",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per device",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = RFTConfig.from_yaml(args.config)
    else:
        config = DEFAULT_CONFIG
    
    # Override with CLI args
    if args.model:
        config.model.model_name = args.model
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.lora_r:
        config.lora.r = args.lora_r
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Check dependencies
    has_unsloth = check_dependencies()
    
    # Determine training data path
    train_path = args.train_data
    val_path = args.val_data
    
    if not train_path:
        # Check for default location
        default_train = Path(config.data.trajectories_dir) / "sft" / "train.jsonl"
        if default_train.exists():
            train_path = str(default_train)
            val_path = str(Path(config.data.trajectories_dir) / "sft" / "val.jsonl")
        else:
            console.print("[red]No training data found![/red]")
            console.print("[yellow]Collect data first with:[/yellow]")
            console.print("  python -m lemonade_bench.rft.collect --provider anthropic")
            sys.exit(1)
    
    # Print config summary
    console.print("\n[bold cyan]═══ RFT Training Configuration ═══[/bold cyan]")
    console.print(f"  Model: {config.model.model_name}")
    console.print(f"  LoRA rank: {config.lora.r}")
    console.print(f"  Learning rate: {config.training.learning_rate}")
    console.print(f"  Epochs: {config.training.num_train_epochs}")
    console.print(f"  Batch size: {config.training.per_device_train_batch_size}")
    console.print(f"  Output: {config.training.output_dir}")
    console.print(f"  Using Unsloth: {has_unsloth}")
    console.print()
    
    # Train
    if has_unsloth:
        train_with_unsloth(config, train_path, val_path)
    else:
        train_with_peft(config, train_path, val_path)


if __name__ == "__main__":
    main()

