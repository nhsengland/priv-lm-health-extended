# Instruction-tuning Dataset
## About
This directory contains the scripts to create an instruction-tuning dataset from existing annotated resources based on MIMIC-III and MIMIC-IV. 

## Usage 
1. Download the resources using `download_datasets.sh` (requires passing PhysioNet credentials through the command line). The script also clones GitHub repositories. Follow the repective instructions in each repository for setup. 
2. Set up file paths in `config.py`. The most important thing to check is that `PROJECT_DIR` matches what is in `download_datasets.sh`.
3. Run `python run.py` to make datasets:
    - A base dataset combining MIMIC-III and IV discharge notes with IDs (arrow).
    - Task-specific datasets with note IDs and labels (arrow).
    - Combined datasets sampled and converted from task-specific datasets, as regular supervised learning input-output pairs (`base`) (jsonl) and instruction tuning format (`chat`) (arrow).

## Example
Load the created instruction tuning data in Python by:
```python
import datasets

dataset = datasets.load_from_disk(
    "<MIMIC_INSTRUCTION_PATH in config.py>/subset_<subset number>_chat",
    split="<split>"
)
```
Example instance:
```python
>>> dataset[0]

{
    'note_id': '<a unique note identifier>',
    'task': 'di',
    'input': '<the full note>',
    'instruction': 'Please write the discharge instruction for the patient based on the clinical document below.\n\n\nHere is the document:\n\n\n',
    'messages': [
        {
            'content': 'You are a helpful assistant that helps with clinical tasks.',
            'role': 'system'
        },
       {
            'content': 'Please write the discharge instruction for the patient based on the clinical document below.\n\n\nHere is the document:\n\n\nName:  ___           ___ No:   ___\n \nAdmission Date:  ___              Discharge Date:   ___\n \nDate of Birth:  ___             Sex:   <the rest of the note goes here>',
            'role': 'user'
        },
        {
            'content': '<the task output (in this case, the discharge instruction) goes here>',
            'role': 'assistant'
        }
    ]
}
```