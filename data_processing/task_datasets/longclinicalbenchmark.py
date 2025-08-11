import os
import json
from config import PROJECT_DIR, NOTE_MATCH_COL
from dataset import TaskDataset
from utils import del_map, aho_corasick_search


class LongClinicalBenchmark(TaskDataset):

    CITATION = """
    Yoon W, Chen S, Gao Y, Zhao Z, Dligach D, Bitterman DS, Afshar M, Miller T. 
    LCD Benchmark: Long Clinical Document Benchmark on Mortality Prediction for Language Models. 
    medRxiv [Preprint]. 2024 Jul 2:2024.03.26.24304920. doi: 10.1101/2024.03.26.24304920. 
    PMID: 38585973; PMCID: PMC10996733.
    """

    def __init__(self, base_dataset):
        super().__init__()
        self.dataset = base_dataset
        self.dataset_dir = os.path.join(
            PROJECT_DIR, "long-clinical-doc", "out_hos_30days_mortality"
        )
        # expected output columns
        self.column_name = "longclinicalbenchmark"

    def process(self):

        for split in ["train", "dev"]:
            with open(os.path.join(self.dataset_dir, f"{split}.json")) as f:
                data = json.load(f)["data"]
            scrunched_texts = [
                d["text"].replace("<cr>", "").lower().translate(del_map) for d in data
            ]
            labels = [d["out_hospital_mortality_30"] for d in data]

        curr_mapper = dict()
        for curr_substring, curr_matches in aho_corasick_search(
            partial_texts=scrunched_texts, full_texts=self.dataset[NOTE_MATCH_COL]
        ).items():
            if len(curr_matches) == 1:
                # create mapping from scrunched original note to label
                idx = scrunched_texts.index(curr_substring)
                label = f"Based on the provided document, the patient's out hospital mortality in 30 days is predicted to be: {labels[idx]}"
                curr_mapper[curr_matches[0]] = label

        self.dataset = self.dataset.map(
            lambda batch: {
                self.column_name: [
                    curr_mapper.get(
                        x,
                        "",
                    )
                    for x in batch[NOTE_MATCH_COL]
                ]
            },
            batched=True,
        )

        self.processed_columns.append(self.column_name)

        self.save_to_disk()
