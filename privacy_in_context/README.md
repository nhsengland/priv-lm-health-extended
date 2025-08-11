# Contextual Integrity - Ambient AI for Clinical Documentation
## About
This directory contains code to run the privacy experiments in work package 3. 

This involves:
1. Fine-tuning models on instruction-tuning data, prepared using code in `data_processing/`.
2. Applying methods aimed at identifying memorisation on trained models.

## Usage 
1. ```pip install -r requirements.txt```
2. Our transcripts are based on modified versions of [aci-bench](https://github.com/wyim/aci-bench). 
    - Clone their repository: ```git clone https://github.com/wyim/aci-bench```
    - Make sure `config.py` has `aci_bench_dir` set to the cloned repository.
3. If you are calling cloud API(s) for generation, make sure to set the corresponding environment variables for each service or pass it as a keyword argument (see `generations.py` for details).
4. Create test cases (modified versions of aci-bench transcripts with associated patient profiles) via `python make_profiles.py`.
5. Generate clinical notes from the created transcripts via `python make_notes.py`.
    - Use command line arguments to specify:
        - `-m`: which model to use for generation; model ID from the official API, model ID from HF hub, or path to local file in transformers format
        - `-p`: whether to add a privacy-related instruction to the prompt
        - `-s`: whether to require the note to follow SOAP structure
        - `-o`: whether to allow an "Other" section in the note
        - `-gte`: whether to follow a generate-then-edit approach, and if so, whether to instruct the model to edit for privacy in a `specific` or `generic` way. Defaults to `none` (generate only, no subsequent edit prompt).
    - e.g. `python make_notes.py -m "meta-llama/Llama-3.1-8B" -p -s`
6. Evaluate output note quality via `python evaluate_notes.py`.
