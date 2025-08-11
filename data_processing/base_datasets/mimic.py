import os
import pandas as pd
import datasets
from dataset import BaseDataset
from utils import del_map
from config import NOTE_MATCH_COL


class MIMIC(BaseDataset):
    def load_or_process(self, mimic_3_path, mimic_4_path, max_length_chars=None):
        self.max_length_chars = max_length_chars
        if os.path.exists(self.dataset_dir):
            self.dataset = datasets.load_from_disk(self.dataset_dir)
        else:
            self.process(mimic_3_path=mimic_3_path, mimic_4_path=mimic_4_path)

    def process(self, mimic_3_path, mimic_4_path):
        """Merge MIMIC-III and MIMIC-IV notes."""
        mimic_3 = pd.read_csv(mimic_3_path, keep_default_na=False)
        # for our purposes, can ignore 'CHARTDATE', 'DESCRIPTION', 'CGID', 'ISERROR'
        mimic_3 = mimic_3[mimic_3["HADM_ID"].str.len() > 0]
        mimic_3 = mimic_3[
            [
                "ROW_ID",
                "SUBJECT_ID",
                "HADM_ID",
                "CHARTTIME",
                "STORETIME",
                "CATEGORY",
                "TEXT",
            ]
        ].copy()
        # approximately match colnames and dtypes to mimic-iv
        mimic_3["CATEGORY"] = mimic_3.CATEGORY.map(
            lambda x: "DS" if x == "Discharge summary" else x
        )
        mimic_3.rename(
            columns={
                "ROW_ID": "note_id",
                "SUBJECT_ID": "subject_id",
                "HADM_ID": "hadm_id",
                "CHARTTIME": "charttime",
                "STORETIME": "storetime",
                "CATEGORY": "note_type",
                "TEXT": "text",
            },
            inplace=True,
        )
        mimic_3["note_id"] = mimic_3["note_id"].astype(str)
        mimic_3["hadm_id"] = mimic_3["hadm_id"].astype(int)
        mimic_3["note_seq"] = -1
        mimic_3["source"] = "MIMIC-III"

        mimic_4 = pd.read_csv(mimic_4_path, keep_default_na=False)
        mimic_4["source"] = "MIMIC-IV"

        self.dataset = datasets.concatenate_datasets(
            [
                datasets.Dataset.from_pandas(mimic_3),
                datasets.Dataset.from_pandas(mimic_4),
            ]
        )

        if self.max_length_chars:
            self.dataset = self.dataset.filter(
                lambda x: len(x["text"]) <= self.max_length_chars
            )
        self.dataset = self.dataset.map(
            lambda x: {NOTE_MATCH_COL: x["text"].lower().translate(del_map)}
        )
        self.save_to_disk()
