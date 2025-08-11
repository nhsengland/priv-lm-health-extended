import os
from base_datasets import MIMIC
from task_datasets import *
from instruction_datasets import *
from config import *
import datasets
import pandas as pd


def sample_synthetic_dataset(num_examples=50, random_state=123):
    """Sample dataset for memorisation experiments with Asclepius models."""
    df = (
        pd.DataFrame(
            datasets.load_dataset(
                "starmpcc/Asclepius-Synthetic-Clinical-Notes", split="train"
            )
        )
        .groupby("task")
        .sample(num_examples, random_state=random_state)
    )
    datasets.Dataset.from_pandas(df).remove_columns("__index_level_0__").save_to_disk(
        os.path.join(MIMIC_INSTRUCTION_PATH, f"sampled_asclepius_{num_examples}")
    )


if __name__ == "__main__":
    mimic = MIMIC(dataset_dir=MIMIC_PATH)
    mimic.load_or_process(
        mimic_3_path=os.path.join(MIMIC_3_DIR, "NOTEEVENTS.csv.gz"),
        mimic_4_path=os.path.join(MIMIC_4_DIR, "discharge.csv.gz"),
        max_length_chars=31000,  # estimate for 8k context length
    )
    for task_dataset in [
        BHC_DI,
        Medication,
        EHRNoteQA,
        SBDH,
        CLIP,
        DiSCQ,
        Phenotype,
        DischargeMe,
        MDACE,
        LongClinicalBenchmark,
    ]:
        task_dataset(mimic.dataset).process()
    make_instruction_dataset_base()
    make_chat_dataset_from_base()
    sample_synthetic_dataset()
