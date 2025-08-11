import os
import json
import pandas as pd
from dataset import TaskDataset
from config import PHYSIONET_DIR


class EHRNoteQA(TaskDataset):

    CITATION = """
    Kweon, S., Kim, J., Kwak, H., Cha, D., Yoon, H., Kim, K. H., Yang, J., Won, S., & Choi, E. (2024). 
    EHRNoteQA: An LLM Benchmark for Real-World Clinical Practice Using Discharge Summaries (version 1.0.1). 
    PhysioNet. https://doi.org/10.13026/acga-ht95.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(PHYSIONET_DIR, "ehr-notes-qa-llms", "1.0.1")
        # expected output columns
        self.column_name = "ehrnoteqa"

    def process(self):

        with open(os.path.join(self.dataset_dir, "EHRNoteQA.jsonl")) as f:
            ehrnoteqa = pd.DataFrame([json.loads(l) for l in f])

        curr_mapper = dict()
        for patient_id, gb in ehrnoteqa.groupby("patient_id"):
            row = gb.iloc[0]
            curr_mapper[patient_id] = json.dumps(
                {
                    "question": row.question,
                    "choices": {
                        "text": [
                            row.choice_A,
                            row.choice_B,
                            row.choice_C,
                            row.choice_D,
                            row.choice_E,
                        ],
                        "label": ["A", "B", "C", "D"],
                    },
                    "answer": row.answer,
                }
            )

        self.dataset = self.dataset.map(
            lambda batch: {
                self.column_name: [curr_mapper.get(x, "") for x in batch["subject_id"]]
            },
            batched=True,
        )

        self.processed_columns.append(self.column_name)

        self.save_to_disk()
