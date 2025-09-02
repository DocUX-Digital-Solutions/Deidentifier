from lib.utils import load_file_as_text, dump_text_as_jsonl

in_file = "medical_note.txt"
texts = 100 * [in_file]
ids = [f"N00{1+i}" for i in range(100)]

out_file = "medical_note.jsonl"

dump_text_as_jsonl(texts, ids, out_file)
