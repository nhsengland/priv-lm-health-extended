from ahocorasick import Automaton
from tqdm.auto import tqdm
import json


def aho_corasick_search(partial_texts, full_texts):
    """Search for substrings from list of full texts."""
    A = Automaton()
    for partial_text in partial_texts:
        A.add_word(partial_text, partial_text)
    A.make_automaton()

    results = {partial_text: [] for partial_text in partial_texts}
    for full_text in tqdm(full_texts):
        for _, found_substring in A.iter(full_text):
            results[found_substring].append(full_text)

    return results


def process_span_annotations(batch, mapper, column_name):
    """Helper for batch processing of span annotations.

    Args:
        batch (LazyBatch or dict): Batch of data where datasets map function is applied.
            Each datum is expected to have 'note_id' and 'text'.
        mapper (dict): Mapping where keys are 'note_id' and values are lists of annotations.
            Each annotation must contain 'start', 'end' indices.
        column_name (str): The name of the column to store the processed annotations.

    Returns:
        dict: Dictionary where the key is the processed column name to be added to the dataset,
            and value is the list of processed annotation in the batch. For consistency, each
            processed annotation is either a json string of the result or an empty string.
    """
    out_annotations = []
    for i, nid in enumerate(batch["note_id"]):
        annotations = mapper.get(nid, "")
        if annotations:
            curr_annotations = []
            for annotation in annotations:
                s, e = annotation["start"], annotation["end"]
                annotation["text"] = batch["text"][i][s:e]
                curr_annotations.append(annotation)
            out_annotations.append(json.dumps(curr_annotations))
        else:
            out_annotations.append("")
    return {column_name: out_annotations}


# create a "scrunched" version of the note for matching
# (only alphanum, all lower, no whitespace)
del_chars = "".join(c for c in map(chr, range(1114111)) if not c.isalnum())
del_map = str.maketrans("", "", del_chars)
