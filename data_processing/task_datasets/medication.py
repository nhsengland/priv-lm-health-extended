import os
import pandas as pd
from utils import process_span_annotations
from dataset import TaskDataset
from config import PHYSIONET_DIR
import re
import glob


class Medication(TaskDataset):

    CITATION = """
    Goel, A., Gueta, A., Gilon, O., Erell, S., & Feder, A. (2023). 
    Medication Extraction Labels for MIMIC-IV-Note Clinical Database (version 1.0.0). 
    PhysioNet. https://doi.org/10.13026/ps1s-ab29.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(
            PHYSIONET_DIR,
            "medication-labels-mimic-note",
            "1.0.0",
            "mimic-iv-note-labels",
        )
        # expected output columns
        self.column_name = "medication"

    def process(self):
        # Iterate over dataset files, get note_id from filename, row-wise extract annotation,
        # then map note_id to annotation
        note_id_to_medications = {
            re.search(r"NoteID-([0-9A-Za-z-]+)", fn).group(1): [
                {
                    "start": row["Start Position"],
                    "end": row["End Position"],
                    "type": row["Annotation"],
                    "group": row["Group"],
                }
                for _, row in pd.read_csv(fn).iterrows()
            ]
            for fn in glob.glob(self.dataset_dir + "/*.csv")
        }

        self.dataset = self.dataset.map(
            process_span_annotations,
            fn_kwargs={
                "mapper": note_id_to_medications,
                "column_name": self.column_name,
            },
            batched=True,
        )
        self.processed_columns.append(self.column_name)

        self.save_to_disk()
