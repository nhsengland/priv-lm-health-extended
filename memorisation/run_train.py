"""
Script to fine-tune LLMs on instruction-tuning data, adapted from: 
https://www.philschmid.de/fine-tune-llms-in-2024-with-trl#3-create-and-prepare-the-dataset.

Assumes datasets have been prepared (see `data_processing/`).

Due to time constraints, the models were trained on two H100s using the author's university compute resources;
however the code (1) should work for fine-tuning smaller models, and (2) can be adapted to use libraries
and quantized checkpoints from e.g. unsloth to fit on a T4.
"""
import argparse
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from data_processing.config import *
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, setup_chat_format
import datasets


def main(args):
    model_id = args.model_id
    max_seq_length = args.max_seq_length
    padding_side = args.padding_side
    pad_token = args.pad_token
    subset_i = args.subset_i

    # name of model (checkpoint) directory
    model_name = f"subset_{subset_i}"
    if "asclepius" in model_id.lower():
        model_name = "asclepius_" + model_name

    dataset_dir = os.path.join(MIMIC_INSTRUCTION_PATH, f"subset_{subset_i}_chat")
    dataset = datasets.load_from_disk(dataset_dir)
    dataset = dataset.remove_columns(
        [c for c in dataset["train"].column_names if c != "messages"]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        cache_dir=CACHE_DIR,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=CACHE_DIR, add_eos_token=True
    )
    tokenizer.padding_side = padding_side
    tokenizer.add_special_tokens({"pad_token": pad_token})
    model.resize_token_embeddings(len(tokenizer))

    model, tokenizer = setup_chat_format(model, tokenizer)

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir=os.path.join(MIMIC_MODELS_PATH, model_name),
        num_train_epochs=5,  # number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        gradient_accumulation_steps=8,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,  # use gradient checkpointing to save memory
        optim="adamw_torch_fused",  # use fused adamw optimizer
        logging_steps=10,  # log every 10 steps
        save_strategy="epoch",  # save checkpoint every epoch
        learning_rate=2e-4,  # learning rate, based on QLoRA paper
        bf16=True,  # use bfloat16 precision
        tf32=True,  # use tf32 precision
        max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",  # use constant learning rate scheduler
        push_to_hub=False,  # push model to hub
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        eval_packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # "/import/nlp/LLMs/Meta-Llama-3-8B/"
    # "meta-llama/Meta-Llama-3-8B"   # base model on HF hub
    # "starmpcc/Asclepius-Llama3-8B" # asclepius model on HF hub

    # for smoke test: "HuggingFaceTB/SmolLM2-135M-Instruct"

    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
        type=str,
        help="Path to local saved model or hub ID",
    )
    parser.add_argument(
        "--max-seq-length",
        default=8000,
        type=int,
        help="Maximum sequence length accepted by model",
    )
    parser.add_argument(
        "--padding-side",
        default="right",
        type=str,
        help="Padding side for tokenizer and model",
    )
    parser.add_argument(
        "--pad-token", default="<pad>", type=str, help="Special token used for padding"
    )
    parser.add_argument(
        "--subset-i",
        default=0,
        type=int,
        help="Instruction-tuning dataset to train on (see `data_processing/instruction_datasets/base.py` for descriptions)",
    )

    args = parser.parse_args()

    main(args)
