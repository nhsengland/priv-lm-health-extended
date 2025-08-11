import os
import json
from pathlib import Path
import pandas as pd
from dataset import TaskDataset
from config import PHYSIONET_DIR


class DiSCQ(TaskDataset):

    CITATION = """
    Lehman, E. (2022). 
    Learning to Ask Like a Physician: a Discharge Summary Clinical Questions (DiSCQ) Dataset (version 1.0). 
    PhysioNet. https://doi.org/10.13026/7v8e-h745.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(PHYSIONET_DIR, "discq", "1.0")
        # expected output columns
        self.column_name = "discq"

    def process(self):

        # includes model-generated questions & answers
        df = pd.read_csv(os.path.join(self.dataset_dir, "human_eval_q.csv"))
        # select annotated subset: understandable, non-trivial, medically significant
        df = df[
            (df.understandable == "Yes")
            & (df.nontrivial == "Yes")
            & (df.medically_significant == "Yes")
        ].reset_index(drop=True)

        # get filename (~note ID) mapping
        i2b2_to_mimic = pd.read_csv(
            os.path.join(self.dataset_dir, "i2b2_to_mimic_map.csv")
        ).dropna()
        i2b2_to_mimic["note_id"] = i2b2_to_mimic["row_num"].map(lambda x: str(int(x)))
        i2b2_to_mimic["file"] = i2b2_to_mimic["file"].map(lambda x: Path(x).name)
        i2b2_to_mimic = {
            i2b2: m for i2b2, m in zip(i2b2_to_mimic["file"], i2b2_to_mimic["note_id"])
        }
        df["note_id"] = df["id"].apply(lambda x: i2b2_to_mimic.get(x, x))

        curr_mapper = dict()
        for note_id, gb in df.groupby("note_id"):
            curr_mapper[note_id] = json.dumps(
                [
                    {
                        "model": row.model,  # some by humans, some by models
                        "trigger": row.trigger,
                        "question": row.question,
                        "answer": row.answer,
                        "sufficient_answer": row.sufficient_answer,  # Fully/Partial/No
                    }
                    for _, row in gb.sort_values(by="ch_trigger_start").iterrows()
                ]
            )

        # Task formulation per paper:
        # QG: '{context}After reading the above EMR, what question do you have about "{trigger}"?'
        # QA: 'With respect to {trigger}, {question}?' - found highest probability span and extend to sentence level

        self.dataset = self.dataset.map(
            lambda batch: {
                self.column_name: [curr_mapper.get(x, "") for x in batch["note_id"]]
            },
            batched=True,
        )

        self.processed_columns.append(self.column_name)

        self.save_to_disk()
