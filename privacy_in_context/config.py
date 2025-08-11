import os

aci_bench_dir = "/datadrive/aci-bench/"  # replace with path to cloned aci-bench repo
aci_bench_data_dir = os.path.join(aci_bench_dir, "data", "challenge_data")
aci_profile_input_path = os.path.join(aci_bench_data_dir, "train_metadata.csv")
aci_transcripts_input_path = os.path.join(aci_bench_data_dir, "train.csv")
profile_path = "profile.json"
transcript_path = "transcripts.json"
results_dir = "/datadrive/wp3_results/"  # specify where to store evaluation results
transcript_converted_path = "transcripts"  # HF dataset name for generated transcript

MODEL_NAMES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
]

sampled_encounter_ids = [
    "D2N006",
    "D2N024",
    "D2N027",
    "D2N029",
    "D2N031",
    "D2N037",
    "D2N051",
    "D2N058",
    "D2N062",
    "D2N065",
]
