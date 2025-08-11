import os
from config import MIMIC_DOWNSTREAM_TASKS_PATH


class BaseDataset:
    def __init__(self, dataset_dir=""):
        self.dataset_dir = dataset_dir
        self.dataset = None

    def process(self):
        raise NotImplementedError()

    def save_to_disk(self):
        self.dataset.save_to_disk(self.dataset_dir)
        self.dataset.cleanup_cache_files()


class TaskDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.processed_columns = []

    def save_to_disk(self):
        for processed_column in self.processed_columns:
            self.dataset.select_columns(["note_id", processed_column]).save_to_disk(
                os.path.join(MIMIC_DOWNSTREAM_TASKS_PATH, processed_column)
            )
        self.dataset.cleanup_cache_files()
