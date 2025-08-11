import os
import json
import datasets
import pandas as pd
from config import (
    MIMIC_INSTRUCTION_PATH,
    DATASET_0_PREFIX,
    DATASET_1_PREFIX,
    DATASET_2_PREFIX,
)
from instruction_datasets import (
    DATASET_0_BASE_NAME,
    DATASET_1_BASE_NAME,
    DATASET_2_BASE_NAME,
)

"""
These functions convert a base instruction-tuning data into OAI chat formats for downstream use
and assumes the user has already ran base.py (`make_instruction_dataset_base').
It currently only supports tasks from experiment setting 1 and 2: 
- di: discharge instruction generation 
- longclinicalbenchmark: classification of 30-day OOH mortality 
"""

task_to_instruction = {
    "di": "Please write the discharge instruction for the patient based on the clinical document below.",
    "longclinicalbenchmark": "Based on the clinical document below, how likely is the given patient's out hospital mortality in 30 days? Please answer with 0 (alive) or 1 (death).",
}


def _make_chat_dataset_from_base(in_path, out_path, seed=123, num_subset_examples=100):
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Base instruction tuning dataset not found. Expected location: {in_path}"
        )
    out_dataset = []
    with open(in_path, "r") as f:
        for line in f:
            d = json.loads(line)
            d["instruction"] = (
                task_to_instruction[d["task"]] + "\n\n\nHere is the document:\n\n\n"
            )
            d["messages"] = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that helps with clinical tasks.",
                },
                {"role": "user", "content": d["instruction"] + d["input"]},
                {"role": "assistant", "content": str(d["output"])},
            ]
            out_dataset.append(d)
    out_dataset = pd.DataFrame(data=out_dataset)
    out_dataset["output"] = out_dataset["output"].astype(str)
    tasks = out_dataset.task.unique()

    out_dataset = datasets.Dataset.from_pandas(out_dataset)
    out_dataset = out_dataset.train_test_split(test_size=200, seed=seed)
    # small subset for further analysis
    sampled_subset = datasets.concatenate_datasets(
        [
            out_dataset["train"]
            .filter(lambda d: d["task"] == task)
            .shuffle(seed=seed)
            .select(range(num_subset_examples))
            for task in tasks
        ]
    )
    valid_test = out_dataset["test"].train_test_split(test_size=0.5, seed=seed)
    datasets.DatasetDict(
        {
            "train": out_dataset["train"],
            "validation": valid_test["train"],
            "test": valid_test["test"],
            "sampled_subset": sampled_subset,
        }
    ).save_to_disk(out_path)


def make_chat_dataset_from_base():
    _make_chat_dataset_from_base(
        in_path=os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_0_BASE_NAME),
        out_path=os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_0_PREFIX + "_chat"),
        num_subset_examples=100,  # 100 per task x 1 task
    )
    _make_chat_dataset_from_base(
        in_path=os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_1_BASE_NAME),
        out_path=os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_1_PREFIX + "_chat"),
        num_subset_examples=50,  # 50 per task x 2 tasks
    )
    _make_chat_dataset_from_base(
        in_path=os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_2_BASE_NAME),
        out_path=os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_2_PREFIX + "_chat"),
        num_subset_examples=50,  # 50 per task x 2 tasks
    )
