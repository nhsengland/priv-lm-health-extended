import os

# directory for the project containing downloaded data;
# this should be the same as the root project directory in `download_datasets.sh`.
PROJECT_DIR = "/datadrive/"

# directory for huggingface library caches
CACHE_DIR = "/import/nlp/jchim/hf-cache/"

# directory for physionet data
PHYSIONET_DIR = os.path.join(PROJECT_DIR, "physionet.org", "files")
# directory containing the downloaded MIMIC-III notes data
MIMIC_3_DIR = os.path.join(PHYSIONET_DIR, "mimiciii", "1.4")
# directory containing the downloaded MIMIC-IV notes data
MIMIC_4_DIR = os.path.join(PHYSIONET_DIR, "mimic-iv-note", "2.2", "note")
# path to store processed MIMIC data in HF datasets (arrow) format.
MIMIC_PATH = os.path.join(PROJECT_DIR, "mimic_processed")
# "/Users/jchim/projects/horsestaple/mimic"

# path to store sub-directories of processed annotated task-specific data in arrow format.
MIMIC_DOWNSTREAM_TASKS_PATH = os.path.join(PROJECT_DIR, "mimic_downstream_tasks")
# "/Users/jchim/projects/horsestaple/mimic_downstream_tasks"

# path to store prepared instruction-tuning data and outputs for each experiment setting
MIMIC_INSTRUCTION_PATH = os.path.join(PROJECT_DIR, "mimic_instruction_tuning")
# "/Users/jchim/projects/horsestaple/mimic_instruction_tuning"

# path to store trained models
MIMIC_MODELS_PATH = os.path.join(PROJECT_DIR, "mimic_instruction_models")

# dataset column containing scrunched text (for matching notes when there is no note ID)
NOTE_MATCH_COL = "note_match_text"

# naming convention for output instruction datasets in each experiment setting
DATASET_0_PREFIX = "subset_0"
DATASET_1_PREFIX = "subset_1"
DATASET_2_PREFIX = "subset_2"
