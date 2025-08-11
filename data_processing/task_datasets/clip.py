import os
import json
import pandas as pd
from dataset import TaskDataset
from config import PHYSIONET_DIR


class CLIP(TaskDataset):

    CITATION = """
    Mullenbach, J., Pruksachatkun, Y., Adler, S., Seale, J., Swartz, J., McKelvey, T. G., Yang, Y., & Sontag, D. (2021). 
    CLIP: A Dataset for Extracting Action Items for Physicians from Hospital Discharge Notes (version 1.0.0). 
    PhysioNet. https://doi.org/10.13026/kw00-z903.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(
            PHYSIONET_DIR, "mimic-iii-clinical-action", "1.0.0"
        )
        # expected output columns
        self.column_name = "clip"

    def process(self):
        document_ids = (
            pd.concat(
                [
                    pd.read_csv(
                        os.path.join(self.dataset_dir, f"{split}_ids.csv"), header=None
                    )
                    for split in ["train", "val"]  # ignore test split here
                ]
            )
            .iloc[:, 0]
            .astype(str)
            .tolist()
        )

        annotations_dir = os.path.join(self.dataset_dir, "character_level")

        curr_mapper = dict()
        for document_id in document_ids:
            with open(os.path.join(annotations_dir, f"{document_id}.json")) as f:
                data = json.load(f)
            # Important to keep text and extract spans from accompanying text
            # because fake names/PHI are inserted for this dataset's annotation
            curr_mapper[document_id] = json.dumps(
                {
                    "text": data["text"],
                    "spans": [
                        {
                            "text": data["text"][span["start"] : span["end"]],
                            "type": span["type"],
                        }
                        for span in data["spans"]
                    ],
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
