def suffix(n):
    return str(n) + (
        "th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )

from typing import Union
import os
def file_exists(path: Union[str, None]) -> bool:
    return  (path is not None and
             os.path.isfile(path) and
             os.path.getsize(path) > 0)

from typing import List, Tuple
import json
def load_input(jsonl_file: str) -> Tuple[List[str], List[str]]:
    input: List[str] = []
    input_ids: List = []
    for jsonl_file in jsonl_file:
        with open(jsonl_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                record = json.loads(line.strip())
                input.append(record["text"])
                input_ids.append(record["id"])

    return input_ids, input



from typing import Optional


def load_file_as_text(file_name: str) -> str:
    with open(file_name, "r", encoding='utf-8') as in_H:
        lines = in_H.readlines()

    return "".join(lines)

def dump_text_as_jsonl(text_files: List[str],
                       ids: Optional[List[str]],
                       out_jsonl: str):
    if ids:
        assert len(ids) == len(text_files)
    else:
        ids = len(text_files) * [None]

    with open(out_jsonl, "w", encoding='utf-8') as out_H:
        for text_file, id in zip(text_files, ids):
            text = load_file_as_text(text_file)
            d = {"text": text, "id": id}
            out_H.write(f"{json.dumps(d)}\n")
