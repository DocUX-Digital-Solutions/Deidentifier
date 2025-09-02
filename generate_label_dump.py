from typing import List, Union

from lib.utils import load_input
from ml_util import docux_logger
from ml_util.encounter_record import EncounterRecord
from ml_util.label_tokens import TokenLabeler
from ml_util.devices import num_cpus, cpu_only
from lib.label_inventory import LABEL_INVENTORY


logger = docux_logger.give_logger()
logger.info(f"LABEL_INVENTORY: ({len(LABEL_INVENTORY)}) {LABEL_INVENTORY}")


def extract_deidentification_labels(inputs: List[Union[str, EncounterRecord]],
                                    output_np: str,
                                    *,
                                    no_cuda: bool = False,
                                    model: str = "StanfordAIMI/stanford-deidentifier-base",
                                    cpu_process_multiplier: int = 2,
                                    gpu_batch_size: int = 32,
                                    ):
    logger.info(f"begin extraction (outer).")
    if isinstance(inputs[0], EncounterRecord):
        inputs = [str(r) for r in inputs]

    only_cpu = cpu_only(no_cuda=no_cuda)
    num_cpu_procssses = 1 if num_cpus == 1 else cpu_process_multiplier * num_cpus
    extract_batch_size = 1 if only_cpu else gpu_batch_size
    logger.info(f"only_cpu: {only_cpu} num_cpu_processes: {num_cpu_procssses} no_cuda: {no_cuda} "
                f"extract_batch_size: {extract_batch_size} num inputs: {len(inputs)}")
    token_labeler = TokenLabeler(labels=LABEL_INVENTORY,
                                 model_name=model,
                                 min_char_per_line=20,
                                 max_char_per_line=512,
                                 sb_num_processes=num_cpu_procssses,
                                 extract_batch_size=extract_batch_size,
                                 )
    token_labeler.get_label_dump(inputs, output_np)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_jsonls', type=str, nargs='+', required=True)
    parser.add_argument('--output_np', type=str, required=True)
    parser.add_argument('--log_file', type=str, default='generate_label_dump.log')
    args = parser.parse_args()

    docux_logger.configure_logger(logger, log_file=args.log_file, supress_stdout=False)

    logger.info(f"starting.")
    _, input = load_input(args.input_jsonls)
    logger.info(f"loaded input.")

    extract_deidentification_labels(input, args.output_np)
