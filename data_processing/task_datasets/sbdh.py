import os
import json
import pandas as pd
from utils import process_span_annotations
from dataset import TaskDataset
from config import PROJECT_DIR


class SBDH(TaskDataset):

    CITATION = """
    Ahsan H, Ohnuki E, Mitra A, Yu H. 
    MIMIC-SBDH: A Dataset for Social and Behavioral Determinants of Health. 
    Proc Mach Learn Res. 2021;149:391-413.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(PROJECT_DIR, "MIMIC-SBDH")
        # expected output columns
        self.column_name = "sbdh"

    def process(self):

        sbdh = pd.read_csv(os.path.join(self.dataset_dir, "MIMIC-SBDH.csv"))
        sbdh_keywords = pd.read_csv(
            os.path.join(self.dataset_dir, "MIMIC-SBDH-keywords.csv")
        )
        sbdh["row_id"] = sbdh["row_id"].astype(str)
        sbdh = sbdh.set_index("row_id")
        sbdh_keywords["row_id"] = sbdh_keywords["row_id"].astype(str)

        """
        https://github.com/hibaahsan/MIMIC-SBDH/blob/main/README.md
        categories:
          "sdoh_community", "sdoh_education", "sdoh_economics",  "sdoh_environment",
          "behavior_alcohol", "behavior_tobacco", "behavior_drug"

        See Section 3.3 for annotation guidelines.
        Community-Present and Community-Absent (0: False, 1: True)
        Education (0: False, 1: True)
        Economics (0: None, 1: True, 2: False)
        Environment (0: None, 1: True, 2: False)
        Alcohol Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
        Tobacco Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
        Drug Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
        """
        # retrieve note-level label
        note_id_to_sbdh = {
            note_id: {k: v for k, v in row.items()} for note_id, row in sbdh.iterrows()
        }

        # retrieve keyword annotations per note
        note_id_to_sbdh_keywords = {
            note_id: [
                {k: v for k, v in row.items() if k != "row_id"}
                for _, row in gb.iterrows()
            ]
            for note_id, gb in sbdh_keywords.groupby("row_id")
        }

        # for our task we select only notes with keywords
        self.dataset = self.dataset.filter(
            lambda x: (x["note_id"] in note_id_to_sbdh)
            and (x["note_id"] in note_id_to_sbdh_keywords)
        ).map(
            process_span_annotations,
            fn_kwargs={
                "mapper": note_id_to_sbdh_keywords,
                "column_name": self.column_name,
            },
            batched=True,
        )

        # merge note-level labels with keyword annotations;
        # flatten into json to work with Arrow backend
        self.dataset = self.dataset.map(
            lambda x: {
                self.column_name: json.dumps(
                    {
                        "keywords": x[self.column_name],
                        "labels": note_id_to_sbdh[x["note_id"]],
                    }
                )
            }
        )

        self.processed_columns.append(self.column_name)

        self.save_to_disk()
