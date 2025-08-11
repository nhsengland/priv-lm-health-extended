# Memorisation
## About
This directory contains code to run the memorisation related experiments in work package 2. This involves:
1. Fine-tuning models on instruction-tuning data, prepared using code in `data_processing/`.
2. Applying methods aimed at identifying memorisation on trained models.

## Usage 
1. (Create a Python 3.10 environment. This is needed for the ai2-olmo package, which is used in the ported memorisation metric implementation from mimir.) 
2. ```pip install -r requirements.txt```
3. Train models using `python run_train.py`. 
4. Run memorisation metrics on trained models using `python run_metrics.py`
5. See if scores from metrics are significantly different for seen vs unseen examples via permutation testing: `python run_results.py`. 