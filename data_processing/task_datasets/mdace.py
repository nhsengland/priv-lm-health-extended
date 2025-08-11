import os
import json
import glob
import pandas as pd
from pathlib import Path
from dataset import TaskDataset
from config import PROJECT_DIR


class MDACE(TaskDataset):

    CITATION = """
    Hua Cheng, Rana Jafari, April Russell, Russell Klopfer, Edmond Lu, Benjamin Striner, and Matthew Gormley. 2023. 
    MDACE: MIMIC Documents Annotated with Code Evidence. In Proceedings of the 61st Annual Meeting of the Association 
    for Computational Linguistics (Volume 1: Long Papers), pages 7534–7550, Toronto, Canada.
    Association for Computational Linguistics.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(
            PROJECT_DIR, "MDACE", "with_text", "gold", "Inpatient"
        )
        # expected output columns
        self.column_name = "mdace"

    def process(self):

        hadm_ids = (
            pd.concat(
                [
                    pd.read_csv(
                        os.path.join(
                            PROJECT_DIR,
                            "MDACE",
                            "splits",
                            "Inpatient",
                            f"MDace-ev-{split}.csv",
                        ),
                        header=None,
                    )
                    for split in ["train", "val"]
                ]
            )
            .iloc[:, 0]
            .tolist()
        )

        curr_mapper = dict()

        for fn in glob.glob(os.path.join(self.dataset_dir, "ICD-9", "1.0", "*.json")):
            hadm_id = int(Path(fn).stem.split("-ICD-")[0])
            if hadm_id not in hadm_ids:
                continue
            with open(fn) as f:
                data9 = json.load(f)
            with open(
                os.path.join(
                    self.dataset_dir, "ICD-10", "1.0", f"{hadm_id}-ICD-10.json"
                )
            ) as f:
                data10 = json.load(f)
            notes9 = [n for n in data9["notes"] if n["category"] == "Discharge summary"]
            notes10 = [
                n for n in data10["notes"] if n["category"] == "Discharge summary"
            ]

            assert data9["hadm_id"] == hadm_id
            assert len(notes9) == len(notes10)
            if len(notes9) != 1:
                continue

            note9, note10 = notes9[0], notes10[0]
            note_id = note9["note_id"]
            assert note_id == note10["note_id"]

            curr_mapper[str(note_id)] = json.dumps(
                {
                    "ICD-9_annotations": note9["annotations"],
                    "ICD-10_annotations": note10["annotations"],
                }
            )

        self.dataset = self.dataset.map(
            lambda batch: {
                self.column_name: [curr_mapper.get(x, "") for x in batch["note_id"]]
            },
            batched=True,
        )

        self.processed_columns.append(self.column_name)

        self.save_to_disk()
