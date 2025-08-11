import os
import json
import pandas as pd
from dataset import TaskDataset
from config import PHYSIONET_DIR


class Phenotype(TaskDataset):

    CITATION = """
    Moseley, E., Celi, L. A., Wu, J., & Dernoncourt, F. (2020). 
    Phenotype Annotations for Patient Notes in the MIMIC-III Database (version 1.20.03). 
    PhysioNet. https://doi.org/10.13026/txmt-8m40.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(
            PHYSIONET_DIR, "phenotype-annotations-mimic", "1.20.03"
        )
        # expected output columns
        self.column_name = "phenotype"

    def process(self):
        df = pd.read_csv(os.path.join(self.dataset_dir, "ACTdb102003.csv"))

        label_cols = [
            "ADVANCED.CANCER",
            "ADVANCED.HEART.DISEASE",
            "ADVANCED.LUNG.DISEASE",
            "ALCOHOL.ABUSE",
            "CHRONIC.NEUROLOGICAL.DYSTROPHIES",
            "CHRONIC.PAIN.FIBROMYALGIA",
            "DEMENTIA",
            "DEPRESSION",
            "DEVELOPMENTAL.DELAY.RETARDATION",
            "NON.ADHERENCE",
            "NONE",
            "OBESITY",
            "OTHER.SUBSTANCE.ABUSE",
            "SCHIZOPHRENIA.AND.OTHER.PSYCHIATRIC.DISORDERS",
            # Unsure -
            # "Indicates ambiguity with regard to one or several of the other indications (including "None")
            # on the part of the note annotator" -> we ignore Unsure entries for our task.
            # 'UNSURE'
        ]

        df.rename(
            columns={
                "HADM_ID": "hadm_id",
                "SUBJECT_ID": "subject_id",
                "ROW_ID": "note_id",
            },
            inplace=True,
        )
        df["note_id"] = df["note_id"].astype(str)
        df = df[df.UNSURE == 0].reset_index(drop=True)
        df = df[["hadm_id", "subject_id", "note_id"] + label_cols]

        curr_mapper = dict()
        for note_id, gb in df.groupby("note_id"):
            out_datum = {
                "subject_id": gb.iloc[0].subject_id.item(),
                "hadm_id": gb.iloc[0].hadm_id.item(),
                "labels": [],
            }
            # each note is doubly annotated (two experts; one phyisician one researcher)
            for i in range(gb.shape[0]):
                present_labels, absent_labels = [], []
                for label_col in label_cols:
                    val = gb.iloc[i][label_col]
                    if val:
                        present_labels.append(label_col)
                    else:
                        absent_labels.append(label_col)
                out_datum["labels"].append(
                    {"present": present_labels, "absent": absent_labels}
                )
            curr_mapper[note_id] = json.dumps(out_datum)
        self.dataset = self.dataset.map(
            lambda batch: {
                self.column_name: [curr_mapper.get(x, "") for x in batch["note_id"]]
            },
            batched=True,
        )

        self.processed_columns.append(self.column_name)

        self.save_to_disk()
