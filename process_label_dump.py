import argparse
from typing import List

from lib.date import DateConverter
from lib.label_processing import LabelPostprocessingModel
from lib.metadata import MetaDataCollection
from lib.utils import file_exists
from ml_util.data import load_json_file
from ml_util.huggingface_interface import TokenClassificationDump
from ml_util.label_tokens import compile_doc_labels
from ml_util.list_file import load_list_file
from ml_util import random_utils
from lib.spoofer import Spoofer
from lib.type_clusters import DEID_Clusters
from lib.utils import load_input

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonls', type=str, nargs='+', required=True)
    parser.add_argument('--label_dump_np', type=str, required=True)
    parser.add_argument('--hospitals', type=str)
    parser.add_argument('--vendors', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--metadata_collection_jsonl', type=str)
    parser.add_argument('--ignore_metadata', action='store_true')
    args = parser.parse_args()

    if args.seed is None:
        random_utils.seed_by_time()
    else:
        random_utils.set_seed(args.seed)

    input_ids, input = load_input(args.input_jsonls)

    if args.ignore_metadata:
        meta_data_collection = None
    else:
        meta_data_collection = MetaDataCollection.from_jsonl(args.metadata_collection_jsonl) \
            if file_exists(args.metadata_collection_jsonl) \
            else MetaDataCollection.from_ids(input_ids)

    # Tweak and tidy the raw NN label output
    if args.hospitals is not None:
        hospitals = load_list_file(args.hospitals)
    else:
        hospitals = meta_data_collection.unique_values_for_field('hospital')

    vendors = load_list_file(args.vendors)
    postprocessor = LabelPostprocessingModel.create(hospitals=hospitals,
                                                    vendors=vendors)

    labels = TokenClassificationDump.build_from_np(args.label_dump_np)
    labels = postprocessor.run(reports=input, predictions=labels)
    # Produce s list of consolidated labels for each document
    compiled_labels = compile_doc_labels(input, labels)

    # don't generate this later...
    date_tally, date_format_tally = DateConverter().tally_dates_and_formats(compiled_labels)

    # DIGDI
    tally_clusters = DEID_Clusters.create(compiled_labels, meta_data_collection, vendors)

    # Need to get the rest of the prior distributions...
    spoofer = Spoofer.create(date_tally=date_tally,
                             date_format_tally=date_format_tally,
                             tally_clusters=tally_clusters)
    spoofed_reports, mappings = spoofer.run(input, compiled_labels)






