from typing import List

from date import DateConverter
from deidentifier_model import LabelPostprocessingModel
from hide_in_plain_sight import Spoofer
from ml_util.huggingface_interface import TokenClassificationDump
from ml_util.label_tokens import compiled_doc_labels
from ml_util.list_file import load_list_file
from ml_util import random_utils

def load_input(input_jsonls: List[str]):
    input: List[str] = []
    for jsonl_file in input_jsonls:
        with open(jsonl_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                input.append(json.loads(line.strip()))

    return input

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonls', type=str, nargs='+', required=True)
    parser.add_argument('--label_dump_np', type=str, required=True)
    parser.add_argument('--hospitals', type=str)
    parser.add_argument('--vendors', type=str)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    if args.seed is None:
        random_utils.seed_by_time()
    else:
        random_utils.set_seed(args.seed)

    # Tweak and tidy the raw NN label output
    hospitals = load_list_file(args.hospitals)
    vendors = load_list_file(args.vendors)
    postprocessor = LabelPostprocessingModel.create(hospitals=hospitals,
                                                    vendors=vendors)

    input = load_input(args.input_jsonls)

    labels = TokenClassificationDump.build_from_np(args.input_dump_np)
    labels = postprocessor.run(reports=input, predictions=labels)
    # Produce list of consolidated labels for each document
    compiled_labels = compiled_doc_labels(input, labels)

    # don't generate this later...
    date_tally, date_format_tally = DateConverter().tally_dates_and_formats(compiled_labels)

    spoofer = Spoofer.create(date_tally=date_tally,
                             date_format_tally=date_format_tally)
    spoofed_reports, mappings = spoofer.run(input, compiled_labels)






