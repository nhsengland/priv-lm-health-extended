import os
import json
import pandas as pd
from utils import del_map, aho_corasick_search
from dataset import TaskDataset
from config import NOTE_MATCH_COL, PHYSIONET_DIR


class BHC_DI(TaskDataset):
    CITATION = """"
    Hegselmann, S., Shen, S., Gierse, F., Agrawal, M., Sontag, D., & Jiang, X. (2024). 
    Medical Expert Annotations of Unsupported Facts in Doctor-Written and LLM-Generated Patient Summaries (version 1.0.0). 
    PhysioNet. https://doi.org/10.13026/a66y-aa53.

    Hegselmann, S., Shen, S. Z., Gierse, F., Agrawal, M., Sontag, D., & Jiang, X. (2024). 
    A Data-Centric Approach To Generate Faithful and High Quality Patient Summaries with Large Language Models. 
    arXiv preprint arXiv:2402.15422.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(
            PHYSIONET_DIR,
            "ann-pt-summ",
            "1.0.0",
        )
        # intermediate temporary columns for matching
        self.bhc_match_col = "match_text_bhc"
        self.di_match_col = "match_text_di"
        # expected output columns
        self.bhc_col = "bhc"
        self.di_col = "di"
        self.ds_input_col = "ds_without_di"
        self.ds_hallucinations_col = "ds_hallucinations"

    def process(self):
        ### Subtask 1 ###
        # input BHC, output DI
        with open(
            os.path.join(
                self.dataset_dir,
                "mimic-iv-note-ext-di-bhc",
                "dataset",
                "train.json",  # process only train split
            )
        ) as f:
            note_di_data = pd.DataFrame([json.loads(l) for l in f])
            note_di_data[self.bhc_match_col] = note_di_data.text.str.translate(
                del_map
            ).str.lower()

        # reverse engineer a mapping from original note -> BHC
        results_mapper = dict()
        for curr_substring, curr_matches in aho_corasick_search(
            partial_texts=note_di_data[self.bhc_match_col].tolist(),
            full_texts=self.dataset[NOTE_MATCH_COL],
        ).items():
            if len(curr_matches) == 1:
                results_mapper[curr_matches[0]] = curr_substring

        # using the original note -> BHC mapping to create a new BHC field
        self.dataset = self.dataset.map(
            lambda batch: {
                self.bhc_match_col: [
                    results_mapper.get(x, "") for x in batch[NOTE_MATCH_COL]
                ]
            },
            batched=True,
        )
        del results_mapper

        # use mapping betweem ((scrunched) BHC and BHC), (BHC and DI) to create DI field + DI matching field
        bhc_bhc_mapper = {
            bhc_s: bhc
            for bhc_s, bhc in zip(
                note_di_data[self.bhc_match_col], note_di_data["text"]
            )
        }
        bhc_di_mapper = {
            bhc_s: di
            for bhc_s, di in zip(
                note_di_data[self.bhc_match_col], note_di_data["summary"]
            )
        }
        self.dataset = self.dataset.map(
            lambda batch: {
                self.bhc_col: [
                    bhc_bhc_mapper.get(x, "") for x in batch[self.bhc_match_col]
                ],
                self.di_col: [
                    bhc_di_mapper.get(x, "") for x in batch[self.bhc_match_col]
                ],
            },
            batched=True,
        )
        del bhc_bhc_mapper, bhc_di_mapper

        self.processed_columns.append(self.bhc_col)
        self.processed_columns.append(self.di_col)

        ### Subtask 2 ###
        # input all discharge note sections prior to DI (incl BHC), output DI
        ## Subtask 2 -
        with open(
            os.path.join(
                self.dataset_dir,
                "mimic-iv-note-ext-di",
                "dataset",
                "train.json",  # use only train split
            )
        ) as f:
            note_di_data = pd.DataFrame([json.loads(l) for l in f])
            note_di_data[self.di_match_col] = note_di_data.summary.str.translate(
                del_map
            ).str.lower()

        # reverse engineer a mapping from scrunched DI to 'full' note input
        di_mapper = {
            di_s: context
            for di_s, context in zip(
                note_di_data[self.di_match_col], note_di_data["text"]
            )
        }
        self.dataset = self.dataset.map(
            lambda batch: {
                self.di_match_col: [
                    x.lower().translate(del_map) for x in batch[self.di_col]
                ]
            },
            batched=True,
        ).map(
            lambda batch: {
                self.ds_input_col: [
                    di_mapper.get(x, "") for x in batch[self.di_match_col]
                ]
            },
            batched=True,
        )
        del di_mapper
        self.processed_columns.append(self.ds_input_col)

        ### Subtask 3 ###
        # dataset contains hallucination annotations of 20 inputs x 5 models = 100 model-generated discharge instructions;
        # but the number of inputs is small and we want to avoid input duplication,
        # so we skip those and focus on the below (100 clinician-annotated discharge instructions).
        # Note there is also a validation split (not included here).

        path_orig = os.path.join(
            self.dataset_dir,  # 100 (BHC, DI) pairs annotated for hallucination spans
            "hallucination_datasets",
            "hallucinations_mimic_di.jsonl",
        )
        path_derived = os.path.join(
            self.dataset_dir,  # same 100 BHC, but with clinician manually improved DIs
            "derived_datasets",
            "hallucinations_mimic_di_cleaned_improved.json",
        )

        # load and merge the two
        with open(path_orig) as f:
            hallucinations_mimic_di_orig = pd.DataFrame([json.loads(l) for l in f])
            hallucinations_mimic_di_orig[self.bhc_match_col] = (
                hallucinations_mimic_di_orig["text"].str.translate(del_map).str.lower()
            )

        with open(path_derived) as f:
            hallucinations_mimic_di_derived = pd.DataFrame([json.loads(l) for l in f])
            hallucinations_mimic_di_derived[
                self.bhc_match_col
            ] = hallucinations_mimic_di_derived.text.str.translate(del_map).str.lower()

        hallucinations_merged = hallucinations_mimic_di_orig.merge(
            hallucinations_mimic_di_derived,
            on=self.bhc_match_col,
            suffixes=("", "_derived"),
        )[["text", "summary", "summary_derived", "labels", self.bhc_match_col]].rename(
            columns={"summary": "before", "summary_derived": "after"}
        )

        del hallucinations_mimic_di_orig, hallucinations_mimic_di_derived

        curr_mapper = {
            match_text: json.dumps(
                # turn into string to work with Arrow
                {
                    "before": gb.before.item(),
                    "after": gb.after.item(),
                    "labels": gb.labels.item(),
                }
            )
            for match_text, gb in hallucinations_merged.groupby(self.bhc_match_col)
        }
        self.dataset = self.dataset.map(
            lambda batch: {
                self.ds_hallucinations_col: [
                    curr_mapper.get(x, "") for x in batch[self.bhc_match_col]
                ]
            },
            batched=True,
        )
        del curr_mapper, hallucinations_merged
        self.processed_columns.append(self.ds_hallucinations_col)

        self.save_to_disk()
