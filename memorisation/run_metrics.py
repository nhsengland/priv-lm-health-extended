"""
Applies memorisation related metrics on trained and finetuned models.

Assumes datasets have been prepared (see `data_processing/`) and models have been trained with `run_train.py`.
"""
import os
from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
from data_processing.config import *
from utils.utils import get_rouge_score
import json
import glob
from tqdm.auto import tqdm
import datasets
import torch
import numpy as np
from mimir import LanguageModel, MinKProbAttack
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM


class Pipeline:
    """Wrapper for text generation."""

    def __init__(
        self,
        model_id,
        model_class=AutoModelForCausalLM,
        padding_side="left",
        device="cuda",
        device_map=None,
        torch_dtype=torch.float16,
        cache_dir=os.environ.get("HF_HOME"),
        low_cpu_mem_usage=True,
    ):
        self.device = device
        self.dtype = torch_dtype
        self.model = model_class.from_pretrained(
            model_id,
            device_map=device_map,
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        if self.device and not device_map:
            self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=False, padding_side=padding_side, cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def generate_batches(
        self,
        data,
        batch_size=48,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.9,
        max_new_tokens=200,
        repetition_penalty=1.0,
        return_full_text=False,
    ):
        for chunk in make_chunk(data, batch_size):
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            prompt_ends = [
                len(_)
                for _ in self.tokenizer.batch_decode(
                    inputs.input_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            ]
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # pad_token_id=self.tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
            )
            outputs = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            if return_full_text:
                for output in outputs:
                    yield output
            else:
                for output, prompt_end in zip(outputs, prompt_ends):
                    yield output[prompt_end:]


def make_chunk(lst, n):
    # Adapted from: https://stackoverflow.com/questions/312443/
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def make_prefix_suffix(texts):
    # Follow Kassem et al., 2024's prefix-suffix cutoff: 33% as prefix, 67% as suffix
    prefixes, suffixes = [], []
    for t in texts:
        idx = len(t) // 3
        prefixes.append(t[:idx])
        suffixes.append(t[idx:])
    return prefixes, suffixes


def compute_probs(model, texts):
    for text in tqdm(texts, position=1):
        yield np.nanmean(model.get_probabilities(text, tokens=None))


def compute_mink_probs(model, texts):
    attk = MinKProbAttack(model)
    for text in tqdm(texts, position=1):
        yield attk._attack(text, probs=None)


def compute_metrics(model_path, is_peft, is_synthetic, texts, prefixes, suffixes):
    model = LanguageModel(name=model_path, is_peft=is_peft)
    probs = list(compute_probs(model=model, texts=texts))
    mink_probs = list(compute_mink_probs(model=model, texts=texts))
    del model

    # we load the model separately again because mimir implementation loads models in 4-bit
    # but we don't necessarily want that when we generate
    pipeline = Pipeline(
        model_path,
        model_class=AutoPeftModelForCausalLM if is_peft else AutoModelForCausalLM,
        # device_map="auto",
        device="cuda",
        padding_side="left" if is_synthetic else "right",
        torch_dtype=torch.bfloat16 if is_peft else torch.float16,
    )
    completions = list(pipeline.generate_batches(prefixes))
    del pipeline
    completions_scores = [
        get_rouge_score(completion, suffix)
        for completion, suffix in zip(completions, suffixes)
    ]
    return (probs, mink_probs, completions, completions_scores)


def main(args):

    base_model_path = args.base_model_path
    synthetic_base_model_path = args.synthetic_base_model_path
    subset_i = args.subset_i

    results_path = os.path.join(
        MIMIC_INSTRUCTION_PATH,
        f"subset_{subset_i}_results.json",
    )

    checkpoint_paths = glob.glob(
        os.path.join(
            MIMIC_MODELS_PATH,
            f"*subset_{subset_i}",
            "checkpoint-*",
        )
    )

    # we know the sampled subset are from the training set
    # and we know the model hasn't seen the test set
    dataset = datasets.load_from_disk(
        os.path.join(MIMIC_INSTRUCTION_PATH, f"subset_{subset_i}_chat")
    )
    seen_dataset = dataset["sampled_subset"]
    unseen_dataset = dataset["test"].select(
        range(min(len(dataset["test"]), len(seen_dataset)))
    )
    unseen_dataset = unseen_dataset.select(
        range(min(len(unseen_dataset), len(seen_dataset)))
    )
    seen_texts, unseen_texts = seen_dataset["input"], unseen_dataset["input"]
    gold_prefixes_seen, gold_suffixes_seen = make_prefix_suffix(texts=seen_texts)
    gold_prefixes_unseen, gold_suffixes_unseen = make_prefix_suffix(texts=unseen_texts)

    # data for probing models that were fine-tuned in two phases (base -> synthetic -> ours);
    # note that here we currently have no `true negatives`
    seen_texts_synthetic = (
        datasets.load_dataset(
            "starmpcc/Asclepius-Synthetic-Clinical-Notes", split="train"
        )
        .shuffle(seed=args.seed)
        .select(range(len(seen_texts)))["note"]
    )
    gold_prefixes_synthetic, gold_suffixes_synthetic = make_prefix_suffix(
        texts=seen_texts_synthetic
    )

    # get probabilities and completions from models
    model_to_probs_seen, model_to_probs_unseen = dict(), dict()
    model_to_mink_probs_seen, model_to_mink_probs_unseen = dict(), dict()
    model_to_completions_seen, model_to_completions_unseen = dict(), dict()
    model_to_completions_scores_seen, model_to_completions_scores_unseen = (
        dict(),
        dict(),
    )
    (
        model_to_probs_synthetic,
        model_to_mink_probs_synthetic,
        model_to_completions_synthetic,
        model_to_completions_scores_synthetic,
    ) = (dict(), dict(), dict(), dict())

    model_paths = [base_model_path, synthetic_base_model_path] + checkpoint_paths
    for model_path in tqdm(model_paths):
        # this is because we only fine-tuned with PEFT;
        # will need to adjust in future cases where the "base" model also involves PEFT
        is_peft = not (
            model_path == base_model_path or model_path == synthetic_base_model_path
        )
        is_synthetic = "asclepius" in model_path.lower()
        if is_peft or is_synthetic:
            # we want to obtain probs & completions of synthetic texts
            # for (1) base models and (2) models that have seen synthetic data
            (
                model_to_probs_synthetic[model_path],
                model_to_mink_probs_synthetic[model_path],
                model_to_completions_synthetic[model_path],
                model_to_completions_scores_synthetic[model_path],
            ) = compute_metrics(
                model_path=model_path,
                is_peft=is_peft,
                is_synthetic=is_synthetic,
                texts=seen_texts_synthetic,
                prefixes=gold_prefixes_synthetic,
                suffixes=gold_suffixes_synthetic,
            )

        # then get probs & completions of notes in our instruction tuning data
        (
            model_to_probs_seen[model_path],
            model_to_mink_probs_seen[model_path],
            model_to_completions_seen[model_path],
            model_to_completions_scores_seen[model_path],
        ) = compute_metrics(
            model_path=model_path,
            is_peft=is_peft,
            is_synthetic=is_synthetic,
            texts=seen_texts,
            prefixes=gold_prefixes_seen,
            suffixes=gold_suffixes_seen,
        )

        (
            model_to_probs_unseen[model_path],
            model_to_mink_probs_unseen[model_path],
            model_to_completions_unseen[model_path],
            model_to_completions_scores_unseen[model_path],
        ) = compute_metrics(
            model_path=model_path,
            is_peft=is_peft,
            is_synthetic=is_synthetic,
            texts=unseen_texts,
            prefixes=gold_prefixes_unseen,
            suffixes=gold_suffixes_unseen,
        )

        # store results
        with open(results_path, "w") as f:
            json.dump(
                {
                    "seen": {
                        "prefix": gold_prefixes_seen,
                        "suffix": gold_suffixes_seen,
                        "completions": model_to_completions_seen,
                        "completions_scores": model_to_completions_scores_seen,
                        "raw": model_to_probs_seen,
                        "mink": model_to_mink_probs_seen,
                    },
                    "unseen": {
                        "prefix": gold_prefixes_unseen,
                        "suffix": gold_suffixes_unseen,
                        "completions": model_to_completions_unseen,
                        "completions_scores": model_to_completions_scores_unseen,
                        "raw": model_to_probs_unseen,
                        "mink": model_to_mink_probs_unseen,
                    },
                    "synthetic": {
                        # prefix and suffix from the synthetic notes
                        "prefix": gold_prefixes_synthetic,
                        "suffix": gold_suffixes_synthetic,
                        "completions": model_to_completions_synthetic,
                        "completions_scores": model_to_completions_scores_synthetic,
                        "raw": model_to_probs_synthetic,
                        "mink": model_to_mink_probs_synthetic,
                    },
                },
                f,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # "/import/nlp/LLMs/Meta-Llama-3-8B/"
    # "meta-llama/Meta-Llama-3-8B"   # base model on HF hub
    # "starmpcc/Asclepius-Llama3-8B" # asclepius model on HF hub

    parser.add_argument(
        "--base-model-path",
        default="/import/nlp/LLMs/Meta-Llama-3-8B/",
        type=str,
        help="Path to local saved base model or hub ID",
    )

    parser.add_argument(
        "--synthetic-base-model-path",
        default="starmpcc/Asclepius-Llama3-8B",
        type=str,
        help="Path to local saved model or hub ID of base model finetuned on synthetic data",
    )

    parser.add_argument(
        "--subset-i",
        default=0,
        type=int,
        help="Instruction-tuning dataset used to train analysed models",
    )

    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="Random seed for sampling synthetic notes",
    )

    args = parser.parse_args()

    main(args)
