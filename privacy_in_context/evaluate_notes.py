import sys
import os
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import get_rouge_score
import argparse
import glob
import re
import json
import pandas as pd
from collections import defaultdict
from config import aci_transcripts_input_path, sampled_encounter_ids, results_dir


def main(args):
    # evaluates all files in the directory that matches the expected pattern (see `make_notes.py')
    notes_dir = os.path.join(args.dir, "notes_*_priv*_soap*_other*_gte*.json")

    results_path = os.path.join(results_dir, "note_quality.json")

    results = defaultdict(list)

    # load gold-standard notes
    df = pd.read_csv(aci_transcripts_input_path)
    for col in df:
        df[col] = df[col].str.strip()
    gold_notes = dict(
        df[df.encounter_id.isin(sampled_encounter_ids)].set_index("encounter_id").note
    )

    for note_path in glob.glob(notes_dir):
        m = re.match(
            ".*/notes_(?P<model_name>.*)_priv(?P<priv>.*)_soap(?P<soap>.*)_other(?P<other>.*)_gte(?P<gte>.*).json",
            note_path,
        )
        model_name = m.group("model_name")
        priv = m.group("priv")
        soap = m.group("soap")
        other = m.group("other")
        gte = m.group("gte")

        with open(note_path, "r") as f:
            note_data = json.load(f)

        for transcript_key, note in note_data.items():
            eles = transcript_key.split("_")
            encounter_id, relation, information_type = (
                eles[0],
                eles[1],
                "_".join(eles[2:]),
            )
            # score note quality wrt to reference (in this case using ROUGE-L)
            score = get_rouge_score(note, gold_notes[encounter_id])

            results["model_name"].append(model_name)
            results["priv"].append(priv)
            results["soap"].append(soap)
            results["other"].append(other)
            results["gte"].append(gte)
            results["relation"].append(relation)
            results["information_type"].append(information_type)
            results["score"].append(score)

        with open(results_path, "w") as f:
            json.dump(dict(results), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Output directory that stores notes to evaluate",
        default="/home/",
    )

    args = parser.parse_args()

    main(args)
