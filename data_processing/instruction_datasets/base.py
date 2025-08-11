import os
import glob
import json
import datasets
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from config import (
    MIMIC_PATH,
    MIMIC_DOWNSTREAM_TASKS_PATH,
    MIMIC_INSTRUCTION_PATH,
    DATASET_0_PREFIX,
    DATASET_1_PREFIX,
    DATASET_2_PREFIX,
    DATASET_3_PREFIX,
)

DATASET_0_BASE_NAME = DATASET_0_PREFIX + "_base.json"
DATASET_1_BASE_NAME = DATASET_1_PREFIX + "_base.json"
DATASET_2_BASE_NAME = DATASET_2_PREFIX + "_base.json"
DATASET_3_BASE_NAME = DATASET_3_PREFIX + "_base.json"


def make_instruction_dataset_base(seed=2024):
    rng = np.random.default_rng(seed)
    dataset = datasets.load_from_disk(MIMIC_PATH)

    subset_to_data = dict()

    for downstream_dir in tqdm(
        glob.glob(
            MIMIC_DOWNSTREAM_TASKS_PATH + "*/"
            if MIMIC_DOWNSTREAM_TASKS_PATH.endswith("/")
            else MIMIC_DOWNSTREAM_TASKS_PATH + "/*/"
        )
    ):
        colname = Path(downstream_dir).stem.lower()
        null_value = ""
        curr_subset = dataset.load_from_disk(downstream_dir).filter(
            lambda x: x[colname] != null_value
        )
        subset_to_data[colname] = curr_subset

    os.makedirs(MIMIC_INSTRUCTION_PATH, exist_ok=True)

    all_sampled_note_ids = []
    note_ids = list(  # 2069
        set(subset_to_data["ds_without_di"]["note_id"]).intersection(
            subset_to_data["longclinicalbenchmark"]["note_id"]
        )
    )

    ##########
    # Subset 0 - 1200 notes for discharge instruction generation
    sampled_note_ids = rng.choice(list(note_ids), size=1200, replace=False)
    subset_di_input = subset_to_data["ds_without_di"].filter(
        lambda x: x["note_id"] in sampled_note_ids
    )
    subset_di_output = subset_to_data["di"].filter(
        lambda x: x["note_id"] in sampled_note_ids
    )
    df = (
        pd.DataFrame(subset_di_input)
        .set_index("note_id")
        .join(pd.DataFrame(subset_di_output).set_index("note_id"))
    )
    with open(os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_0_BASE_NAME), "w") as f:
        # we store one task annotation per line
        for note_id, row in df.iterrows():
            f.write(
                json.dumps(
                    {
                        "note_id": note_id,
                        "task": "di",
                        "input": row.ds_without_di,
                        "output": str(row.di),
                    }
                )
                + "\n"
            )
    all_sampled_note_ids.extend(sampled_note_ids)

    ##########
    # Subset 1 - 600 doubly annotated notes (discharge summary-DI -> DI and 30 day OOH mortality) - 1200 total
    # 10:1:1 train:val:test

    sampled_note_ids = rng.choice(list(note_ids), size=600, replace=False)

    subset_di_input = subset_to_data["ds_without_di"].filter(
        lambda x: x["note_id"] in sampled_note_ids
    )
    subset_di_output = subset_to_data["di"].filter(
        lambda x: x["note_id"] in sampled_note_ids
    )
    subset_lcd = subset_to_data["longclinicalbenchmark"].filter(
        lambda x: x["note_id"] in sampled_note_ids
    )

    df = (
        pd.DataFrame(subset_di_input)
        .set_index("note_id")
        .join(pd.DataFrame(subset_di_output).set_index("note_id"))
        .join(pd.DataFrame(subset_lcd).set_index("note_id"))
    )
    # we consider further processing (e.g. add additional PII) later
    with open(os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_1_BASE_NAME), "w") as f:
        # we store one task annotation per line
        for note_id, row in df.iterrows():
            f.write(
                json.dumps(
                    {
                        "note_id": note_id,
                        "task": "di",
                        "input": row.ds_without_di,
                        "output": str(row.di),
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "note_id": note_id,
                        "task": "longclinicalbenchmark",
                        "input": row.ds_without_di,
                        "output": str(row.longclinicalbenchmark),
                    }
                )
                + "\n"
            )
    all_sampled_note_ids.extend(sampled_note_ids)

    ##########
    # Subset 2 - 600 singly annotated notes for each task (DS-DI -> DI OR 30 day OOH mortality) - 1200 total
    # 10:1:1 train:val:test

    num_examples = 600

    # sample from long clinical benchmark - must have valid input doc from ds_without_di subset
    lcd_note_ids = set(
        rng.choice(
            list(
                set(subset_to_data["ds_without_di"]["note_id"]).intersection(
                    subset_to_data["longclinicalbenchmark"]["note_id"]
                )
            ),
            num_examples,
            replace=False,
        )
    )
    lcd_subset_inputs = subset_to_data["ds_without_di"].filter(
        lambda x: x["note_id"] in lcd_note_ids
    )
    lcd_subset = subset_to_data["longclinicalbenchmark"].filter(
        lambda x: x["note_id"] in lcd_note_ids
    )

    # sample discharge instruction generation subset
    di_subset_note_ids = set(
        rng.choice(
            list(set(subset_to_data["ds_without_di"]["note_id"]) - lcd_note_ids),
            num_examples,
            replace=False,
        )
    )
    assert not lcd_note_ids.intersection(di_subset_note_ids)
    di_subset_inputs = subset_to_data["ds_without_di"].filter(
        lambda x: x["note_id"] in di_subset_note_ids
    )
    di_subset = subset_to_data["di"].filter(
        lambda x: x["note_id"] in di_subset_note_ids
    )

    # process both and convert into flat dataset
    out_data = []
    df1 = (
        pd.DataFrame(lcd_subset_inputs)
        .set_index("note_id")
        .join(pd.DataFrame(lcd_subset).set_index("note_id"))
    )
    for note_id, row in df1.iterrows():
        out_data.append(
            {
                "note_id": note_id,
                "task": "longclinicalbenchmark",
                "input": row.ds_without_di,
                "output": row.longclinicalbenchmark,
            }
        )
    df2 = (
        pd.DataFrame(di_subset_inputs)
        .set_index("note_id")
        .join(pd.DataFrame(di_subset).set_index("note_id"))
    )
    for note_id, row in df2.iterrows():
        out_data.append(
            {
                "note_id": note_id,
                "task": "di",
                "input": row.ds_without_di,
                "output": row.di,
            }
        )

    assert len(out_data) == num_examples * 2, str(len(out_data))

    all_sampled_note_ids.extend(sampled_note_ids)
    with open(os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_2_BASE_NAME), "w") as f:
        for out_d in out_data:
            f.write(json.dumps(out_d) + "\n")

    ##########
    # Subset 3 -
    # 120 each for 9 tasks, 100 for ds_hallucinations (only 100 total in training set; we exclude heldout test set here)
    # 1180

    # 10:1:1 train:val:test
    num_examples = 120

    out_data = []

    sampled_data = dict()
    sampled_note_ids = []

    for task_dataset_name in [
        # roughly in order of data availability
        # MIMIC-III
        "mdace",
        "sbdh",
        "clip",
        "phenotype",
        # MIMIC-IV
        "ds_hallucinations",
        "ehrnoteqa",
        "medication",
        "longclinicalbenchmark",
        "ds_without_di",
        "bhc",
    ]:
        curr_task_subset = (
            subset_to_data[task_dataset_name]
            .filter(lambda x: x["note_id"] not in sampled_note_ids)
            .shuffle(rng)
            .select(range(min(num_examples, len(subset_to_data[task_dataset_name]))))
        )
        sampled_note_ids.extend(set(curr_task_subset["note_id"]))
        sampled_data[task_dataset_name] = curr_task_subset

    all_sampled_note_ids.extend(sampled_note_ids)

    # convert into input-output
    sampled_orig_dataset = dataset.filter(lambda x: x["note_id"] in sampled_note_ids)
    sampled_di_note_ids = set(
        sampled_data["ds_without_di"]["note_id"] + sampled_data["bhc"]["note_id"]
    )

    note_id_to_orig = {d["note_id"]: d["text"] for d in sampled_orig_dataset}
    note_id_to_di = {
        d["note_id"]: d["di"]
        for d in subset_to_data["di"]
        if d["note_id"] in sampled_di_note_ids
    }

    for task_dataset_name, task_data in sampled_data.items():
        for d in task_data:
            note_id = d["note_id"]
            if task_dataset_name == "clip":
                # note: there is already fake PII injected - no need to process
                curr_annotations = json.loads(d[task_dataset_name])
                input_text = curr_annotations["text"]
                out = curr_annotations["spans"]
            elif task_dataset_name in ["ds_without_di", "bhc"]:
                input_text = note_id_to_di[note_id]
                out = d[task_dataset_name]
            else:
                input_text = note_id_to_orig[note_id]
                if task_dataset_name in [
                    "mdace",
                    "sbdh",
                    "ds_hallucinations",
                    "ehrnoteqa",
                    "medication",
                ]:
                    out = json.loads(d[task_dataset_name])
                elif task_dataset_name == "phenotype":
                    out = json.loads(d[task_dataset_name])["labels"]
                else:
                    out = d[task_dataset_name]

            out_data.append(
                {
                    "note_id": note_id,
                    "task": task_dataset_name,
                    "input": input_text,
                    "output": out,
                }
            )

    with open(os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_3_BASE_NAME), "w") as f:
        for out_d in out_data:
            f.write(json.dumps(out_d) + "\n")

    # keep track of which notes need extra processing for PII (not explored in project)
    with open(
        os.path.join(MIMIC_INSTRUCTION_PATH, DATASET_3_PREFIX + "_note_id.json"), "w"
    ) as f:
        json.dump(
            {
                "do_inject": list(
                    set(all_sampled_note_ids) - set(sampled_data["clip"]["note_id"])
                ),
                "skip_inject": sampled_data["clip"]["note_id"],
            },
            f,
        )
