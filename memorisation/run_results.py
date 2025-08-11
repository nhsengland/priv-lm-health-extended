"""
Example code to analyse results from applying memorisation-related metrics. 

Assumes:
- datasets have been prepared (see `data_processing/')
- models have been trained with `run_train.py'
- results were obtained with `run_metrics.py'
"""
import os
from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from data_processing.config import *
import argparse
import re
import json
import numpy as np
from mlxtend.evaluate import permutation_test
import logging

logger = logging.getLogger(__name__)


def main(args):

    base_model_path = args.base_model_path
    synthetic_base_model_path = args.synthetic_base_model_path
    ref_models = [base_model_path, synthetic_base_model_path]
    do_run_synthetic = args.synthetic
    subset_i = args.subset_i

    results_path = os.path.join(
        MIMIC_INSTRUCTION_PATH,
        f"subset_{subset_i}_results.json",
    )

    with open(results_path, "r") as f:
        results = json.load(f)

    for metric in ["completions_scores", "raw", "mink"]:

        logger.info(f"***{metric}***")

        model_ids = []
        for model_id in results["seen"][metric]:
            if model_id in ref_models:
                continue
            if "asclepius" in model_id:
                if do_run_synthetic:
                    model_ids.append(model_id)
            else:
                if not do_run_synthetic:
                    model_ids.append(model_id)

        model_ids = sorted(
            model_ids, key=lambda x: int(re.match(".*checkpoint-(\d{1,5})", x).group(1))
        )

        for model_id in model_ids:
            seen_scores = np.array(results["seen"][metric][model_id])
            unseen_scores = np.array(results["unseen"][metric][model_id])
            # scores from reference model (base)
            base_scores_seen = np.array(results["seen"][metric][base_model_path])
            base_scores_seen[base_scores_seen == 0] = 1e-08  # avoid zero division
            base_scores_unseen = np.array(results["unseen"][metric][base_model_path])
            base_scores_unseen[base_scores_unseen == 0] = 1e-08

            # compute calibrated versions of scores using scores from reference models
            if metric == "completions_scores":
                calibrated_seen_scores = seen_scores / base_scores_seen
                calibrated_unseen_scores = unseen_scores / base_scores_unseen
            else:  # based on log likelihoods
                calibrated_seen_scores = seen_scores - base_scores_seen
                calibrated_unseen_scores = unseen_scores - base_scores_unseen

            p_value = permutation_test(
                seen_scores,
                unseen_scores,
                method="approximate",
                num_rounds=10000,
                seed=args.seed,
            )

            p_value_calibrated = permutation_test(
                calibrated_seen_scores,
                calibrated_unseen_scores,
                method="approximate",
                num_rounds=10000,
                seed=args.seed,
            )

            if do_run_synthetic:
                # scores from reference model (synthetic)
                synthetic_base_scores_seen = np.array(
                    results["seen"][metric][synthetic_base_model_path]
                )
                synthetic_base_scores_unseen = np.array(
                    results["unseen"][metric][synthetic_base_model_path]
                )
                synthetic_base_scores_seen[synthetic_base_scores_seen == 0] = 1e-08
                synthetic_base_scores_unseen[synthetic_base_scores_unseen == 0] = 1e-08

                if metric == "completions_scores":
                    synthetic_calibrated_seen_scores = (
                        seen_scores / synthetic_base_scores_seen
                    )
                    synthetic_calibrated_unseen_scores = (
                        unseen_scores / synthetic_base_scores_unseen
                    )
                else:
                    synthetic_calibrated_seen_scores = (
                        seen_scores - synthetic_base_scores_seen
                    )
                    synthetic_calibrated_unseen_scores = (
                        unseen_scores - synthetic_base_scores_unseen
                    )

                p_value_calibrated_synthetic = permutation_test(
                    synthetic_calibrated_seen_scores,
                    synthetic_calibrated_unseen_scores,
                    method="approximate",
                    num_rounds=10000,
                    seed=args.seed,
                )

                logger.info(
                    f"{model_id} - {p_value:.3f} - {p_value_calibrated:.3f} - {p_value_calibrated_synthetic:.3f}"
                )

            else:
                logger.info(f"{model_id} - {p_value:.3f} - {p_value_calibrated:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "-s",
        "--synthetic",
        action="store_true",
        help="Analyse models that were first trained on synthetic data. If False, this script will only analyse models directly fine-tuned from the base model.",
    )

    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="Random seed for permutation testing",
    )

    args = parser.parse_args()

    main(args)
