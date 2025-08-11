import os
import json
import pandas as pd
from dataset import TaskDataset
from config import PHYSIONET_DIR


class DischargeMe(TaskDataset):

    CITATION = """
    Xu, J. (2024). 
    Discharge Me: BioNLP ACL'24 Shared Task on Streamlining Discharge Documentation (version 1.3). 
    PhysioNet. https://doi.org/10.13026/0zf5-fx50.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(PHYSIONET_DIR, "discharge-me", "1.3")
        # expected output columns
        self.column_name = "dischargeme"

    def process(self):

        processed_splits = []

        # skip test data in current processing
        for split in ["train", "valid"]:

            split_dir = os.path.join(self.dataset_dir, split)

            # for our project we limit ourselves to individuals with a single stay
            curr_edstays = pd.read_csv(
                os.path.join(split_dir, "edstays.csv.gz"), keep_default_na=False
            )
            curr_edstays["num_stays_of_subject"] = curr_edstays.groupby(
                "subject_id"
            ).transform("size")
            curr_edstays = curr_edstays[
                curr_edstays["num_stays_of_subject"] == 1
            ].reset_index(drop=True)
            # in this simplified subset, each row now corresponds to a unique subject_id and its unique hadm_id and stay_id

            # get note_id to hadm_id mapping from target df
            curr_target = pd.read_csv(
                os.path.join(split_dir, "discharge_target.csv.gz"),
                keep_default_na=False,
            )
            curr_target = curr_target[curr_target.hadm_id.isin(curr_edstays.hadm_id)]
            note_id_to_hadm_id = {
                n: h for n, h in zip(curr_target["note_id"], curr_target["hadm_id"])
            }
            hadm_id_to_note_id = {h: n for n, h in note_id_to_hadm_id.items()}
            del curr_target

            # get triage data, including chief complaints
            curr_triage = pd.read_csv(
                os.path.join(split_dir, "triage.csv.gz"), keep_default_na=False
            )
            stay_id_to_triage = dict()
            for stay_id, gb in curr_triage.groupby("stay_id"):
                stay_id_to_triage[stay_id] = {
                    k: v.item() for k, v in dict(gb).items() if k != "stay_id"
                }
            del curr_triage

            # get diagnosis codes
            stay_id_to_codes = {
                stay_id: [
                    {"icd_code": c, "icd_version": v, "icd_title": t}
                    for c, v, t in zip(
                        gb["icd_code"], gb["icd_version"], gb["icd_title"]
                    )
                ]
                for stay_id, gb in pd.read_csv(
                    os.path.join(split_dir, "diagnosis.csv.gz"), keep_default_na=False
                ).groupby("stay_id")
            }

            # get radiology reports (each subject has at least 1)
            curr_radiology_report = pd.read_csv(
                os.path.join(split_dir, "radiology.csv.gz"), keep_default_na=False
            )
            hadm_id_to_rr = dict()
            for hadm_id, gb in curr_radiology_report[
                curr_radiology_report.hadm_id.isin(curr_edstays.hadm_id)
            ].groupby("hadm_id"):
                hadm_id_to_rr[hadm_id] = [
                    {
                        k: v
                        for k, v in dict(row).items()
                        if k != "subject_id" and k != "hadm_id"
                    }
                    for _, row in gb.sort_values(by="charttime").iterrows()
                ]
            del curr_radiology_report

            curr_edstays["ds_note_id"] = curr_edstays.hadm_id.map(hadm_id_to_note_id)
            curr_edstays["radiology_reports"] = curr_edstays.hadm_id.map(hadm_id_to_rr)
            curr_edstays["diagnosis"] = curr_edstays.stay_id.map(stay_id_to_codes)
            curr_edstays["triage"] = curr_edstays.stay_id.map(stay_id_to_triage)

            curr_edstays = curr_edstays[
                [
                    "intime",
                    "outtime",
                    "gender",
                    "race",
                    "arrival_transport",
                    "disposition",
                    "diagnosis",
                    "ds_note_id",
                    "triage",
                    "radiology_reports",
                ]
            ].rename(
                columns={
                    "intime": "ed_intime",
                    "outtime": "ed_outtime",
                    "disposition": "ed_disposition",
                }
            )

            curr_edstays["dischargeme_split"] = split

            processed_splits.append(curr_edstays)

        curr_edstays = pd.concat(processed_splits)
        curr_edstays = curr_edstays.set_index("ds_note_id")

        curr_mapper = {
            str(note_id): json.dumps(dict(row))
            for note_id, row in curr_edstays.iterrows()
        }

        self.dataset = self.dataset.map(
            lambda batch: {
                self.column_name: [curr_mapper.get(x, "") for x in batch["note_id"]]
            },
            batched=True,
        )

        self.processed_columns.append(self.column_name)

        self.save_to_disk()
