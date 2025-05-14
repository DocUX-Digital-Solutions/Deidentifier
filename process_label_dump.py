from typing import List

from deidentifier_model import LabelPostprocessingModel
from ml_util.huggingface_interface import TokenClassificationDump
from ml_util.label_tokens import compiled_doc_labels
from ml_util.list_file import load_list_file

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonls', type=str, nargs='+', required=True)
    parser.add_argument('--label_dump_np', type=str, required=True)
    parser.add_argument('--hospitals', type=str)
    parser.add_argument('--vendors', type=str)
    args = parser.parse_args()

    input: List[str] = []
    for jsonl_file in args.input_jsonls:
        with open(jsonl_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                input.append(json.loads(line.strip()))

    labels = TokenClassificationDump.build_from_np(args.input_dump_np)

    hospitals = load_list_file(args.hospitals)
    vendors = load_list_file(args.vendors)
    postprocessor = LabelPostprocessingModel.create(hospitals=hospitals,
                                                    vendors=vendors)
    labels = postprocessor.run(reports=input, predictions=labels)
    compiled_labels = compiled_doc_labels(input, labels)




