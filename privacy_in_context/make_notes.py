"""
Based on (modified) transcripts, generate notes.

Usage:
    $ python make_notes.py -m "meta-llama/Llama-3.1-8B" -p -s

External dependencies:
    - By default this will try to call gpt-4o; requires OpenAI API key in env (but can switch to local model)
    - Assumes aci-bench data exist
"""
import os
import json
import datasets
from tqdm.auto import tqdm
import argparse
import logging
from pathlib import Path
from generations import LLMGenerator
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoteGenerator:
    def __init__(self, args):
        self.args = args
        if args.model_name.startswith("claude-"):
            provider = "anthropic"
        else:
            provider = "huggingface"
        self.llm = LLMGenerator(provider=provider, model_name=args.model_name)

    def build_generate_note_prompt(self):
        prompt = "Please generate a clinical note from this transcript."
        if self.args.priv:
            prompt += " Only include information relevant to the patient's care."
        if self.args.soap:
            prompt += (
                "\n\nRequirements:\n"
                "- The note should contain Subjective, Objective, Assessment, and Plan sections.\n"
            )
            if self.args.other:
                prompt += "- Information that do not fall into these sections should go in an (optional) Other section.\n"
            prompt += "- Within each section there can be further sub-headings."
        prompt += "\n\n"
        return prompt

    def build_edit_note_prompt(self, is_specific=True):
        if is_specific:
            prompt = "Please edit this note so that it only includes information relevant to the patient's care."
        else:
            prompt = "Please edit this note to preserve privacy."
        prompt += "\n\n"
        return prompt

    def run_prompt(self, example_id, prompt):
        text, stop_reason = self.llm.generate(prompt)
        if stop_reason not in ["stop", "end_turn", "eos", "EOS"]:
            logger.warning(f"{example_id} {stop_reason}")
        return text


def main(args):
    clinical_note_path = os.path.join(
        args.dir,
        f"notes_{Path(args.model_name).name}_priv{args.priv}_soap{args.soap}_other{args.other}_gte{args.generate_then_edit}.json",
    )
    if os.path.exists(clinical_note_path):
        with open(clinical_note_path, "r") as f:
            out_data = json.load(f)
    else:
        out_data = dict()

    logger.info(clinical_note_path)

    note_generator = NoteGenerator(args)
    dataset = datasets.load_from_disk(transcript_converted_path, split="train")

    for datum in tqdm(dataset, position=0, leave=True):
        transcript_key = "_".join(
            [datum["encounter_id"], datum["relation"], datum["information_type"]]
        )

        if transcript_key in out_data:
            continue

        # Generate initial note
        prompt = note_generator.build_generate_note_prompt() + datum["transcript"]
        generated_note = note_generator.run_prompt(transcript_key, prompt)

        # If generate_then_edit is specified, apply the edit step accordingly
        if args.generate_then_edit != "none":
            is_specific = args.generate_then_edit == "specific"
            edit_prompt = (
                note_generator.build_edit_note_prompt(is_specific=is_specific)
                + generated_note
            )
            final_note = note_generator.run_prompt(transcript_key, edit_prompt)
        else:
            final_note = generated_note

        out_data[transcript_key] = final_note

        with open(clinical_note_path, "w") as f:
            json.dump(out_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument(
        "-p",
        "--priv",
        action="store_true",
        help="If True, adds privacy related instruction to prompt.",
    )
    parser.add_argument(
        "-s",
        "--soap",
        action="store_true",
        help="If True, adds note structure instruction to prompt.",
    )
    parser.add_argument(
        "-o",
        "--other",
        action="store_true",
        help="If True, adds instruction for optional Other section (requires --soap).",
    )

    parser.add_argument(
        "-gte",
        "--generate_then_edit",
        type=str,
        choices=["none", "specific", "generic"],
        default="none",
        help="Type of edit to apply after generation. 'specific' edits for patient-relevant info, 'generic' edits for privacy.",
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Output directory to store files",
        default="/home/",
    )

    args = parser.parse_args()

    # Other section can only be added if specifying SOAP
    if args.other and not args.soap:
        parser.error("The --other argument requires --soap to be enabled")

    main(args)
