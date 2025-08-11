"""
Based on aci-bench, adds personal data to profiles then uses them to modify transcripts.

Usage:
    $ python make_profiles.py

External dependencies:
    - By default this will try to call gpt-4o; requires OpenAI API key in env (but can switch to local model)
    - Assumes aci-bench data exist
"""

import os
import json
import json_repair
import datasets
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict
from collections import defaultdict
import logging
from generations import LLMGenerator
from prompts import Prompts
from personal_information import PersonalInformation
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfileGenerator:
    def __init__(self, temperature: float = 0.0):
        self.llm = LLMGenerator(
            provider="openai", model_name="gpt-4o-2024-08-06", temperature=temperature
        )

    def _standardise_profile(self, profile: dict) -> dict:
        k_mapper = {}
        for k in profile:
            # heuristic - sometimes LLMs will rewrite underscore-separated keys into well-formed text
            if k == "lifestyle, habits, and recreational activities":
                new_k = "lifestyle_habits_and_recreational_activities"
                k_mapper[k] = new_k
            else:
                new_k = k.replace(" ", "_")
                k_mapper[k] = new_k
        for k_old, k_new in k_mapper.items():
            profile[k_new] = profile[k_old]
        return profile

    def generate_profile(self, patient_data: pd.Series, max_tokens=2000) -> Dict:
        prompt = Prompts.PROFILE.format(
            f_name=patient_data.patient_firstname,
            l_name=patient_data.patient_familyname,
            gender=patient_data.patient_gender,
            age=patient_data.patient_age,
            cc=patient_data.cc,
            second_complaint="none"
            if pd.isna(patient_data["2nd_complaints"])
            else patient_data["2nd_complaints"],
        )

        response_text, finish_reason = self.llm.generate(prompt, max_tokens=max_tokens)

        if finish_reason != "stop":
            logger.warning(
                f"Generation incomplete for patient {patient_data.encounter_id}: {finish_reason}"
            )

        return {
            **{k: v for k, v in dict(patient_data).items() if k != "encounter_id"},
            "profile_input": prompt,
            "profile_output": response_text,
        }

    def generate_modified_transcripts(
        self, profile: dict, transcript: str, relationship: str, max_tokens: int = 2000
    ) -> Dict[str, str]:
        """Given transcript and relationship, generate modified versions over all information types."""
        modified_transcripts = {}

        for category in PersonalInformation.info_type:
            prompt = Prompts.INJECT_INFO.format(
                profile=str(profile),
                transcript=transcript.replace("[doctor]", "[speaker1]").replace(
                    "[patient]", "[speaker2]"
                ),
                category=category,
                relationship=relationship,
            )
            response_text, finish_reason = self.llm.generate(
                prompt, max_tokens=max_tokens
            )

            if finish_reason != "stop":
                logger.warning(
                    f"Modification incomplete for category {category}: {finish_reason}"
                )

            modified_transcripts[category] = str(response_text)

        return modified_transcripts


def convert_transcript_to_dataset(transcript_path: str, out_path: str):
    """Save modified transcripts into datasets format for easier handling."""
    out_data = defaultdict(list)

    with open(transcript_path, "r") as f:
        data = json.load(f)

    for relation in PersonalInformation.relationship_type:
        for encounter_id, encounter_datum in data.items():
            for information_type, transcript in encounter_datum[relation].items():
                out_data["encounter_id"].append(encounter_id)
                out_data["relation"].append(relation)
                out_data["information_type"].append(information_type)
                out_data["transcript"].append(transcript)
    datasets.Dataset.from_pandas(pd.DataFrame(out_data)).save_to_disk(out_path)


def main():
    # Load data;
    # previously selected based on stratified random sampling without replacement on `dataset`, `patient_age` (bin by decade)
    sampled_patients = pd.read_csv(aci_profile_input_path)
    sampled_patients = sampled_patients[
        sampled_patients.encounter_id.isin(sampled_encounter_ids)
    ].reset_index(drop=True)

    df = pd.read_csv(aci_transcripts_input_path)
    df = df[df.encounter_id.isin(sampled_encounter_ids)].reset_index(drop=True)

    profile_generator = ProfileGenerator()

    # Process profiles (if processed file not exist)
    if not os.path.exists(profile_path):
        patient_profile_data = {}
        for _, row in tqdm(
            sampled_patients.iterrows(), total=sampled_patients.shape[0]
        ):
            profile_data = profile_generator.generate_profile(row)
            patient_profile_data[row.encounter_id] = profile_data
            with open(profile_path, "w") as f:
                json.dump(patient_profile_data, f)
    else:
        with open(profile_path, "r") as f:
            patient_profile_data = json.load(f)

    # Process transcripts (if generated transcripts not exist)
    if not os.path.exists(transcript_path):
        logger.info(f"Running for {transcript_path}")

        for encounter_id, datum in tqdm(
            patient_profile_data.items(), position=0, leave=True
        ):
            profile = json_repair.loads(datum["profile_output"])
            profile = profile_generator._standardise_profile(profile)

            demographic_cols = [
                "first_name",
                "last_name",
                "gender",
                "age",
                "chief_complaint",
                "2nd_complaint",
            ]
            transcript = df[df.encounter_id == encounter_id].dialogue.item()

            # Make sure data shown in context only contains fields we want
            selected_profile = {
                k: v
                for k, v in profile.items()
                if k in PersonalInformation.info_type or k in demographic_cols
            }

            # Generate modifications for relationships
            for relationship in tqdm(
                PersonalInformation.relationship_type, position=1, leave=False
            ):
                modified_transcripts = profile_generator.generate_modified_transcripts(
                    profile=selected_profile,
                    transcript=transcript,
                    relationship=relationship,
                )
                patient_profile_data[encounter_id][relationship] = modified_transcripts
                with open(transcript_path, "w") as f:
                    json.dump(patient_profile_data, f)

        convert_transcript_to_dataset(
            transcript_path=transcript_path, out_path=transcript_converted_path
        )


if __name__ == "__main__":
    main()
