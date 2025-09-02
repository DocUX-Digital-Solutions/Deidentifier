from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import json


@dataclass(frozen=True)
class RecordMetaData:
    id: str

    hcw_given: str
    hcw_middle: Optional[str]
    hcw_surname: str

    patient_given: str
    patient_preferred: Optional[str]
    patient_middle: Optional[str]
    patient_surname: str
    patient_sex: str
    patient_age: int

    hospital_nane: str
    hospital_department: str
    hospital_university: Optional[str]

    unique_values: Optional[List[str]]

    phone_values: Optional[List[str]]

    list_fields = ('unique_values', 'phone_values')

    def __post_init__(self):
        if self.patient_sex not in ('male', 'female'):
            raise ValueError(f"invalid sex value: {self.patient_sex}")

    @classmethod
    def from_api(cls,
                 id: str,
                 api_results):
        raise NotImplementedError
        # Need details of API...

    def to_dict(self) -> Dict:
        out = {k: v for k, v in self.__dict__.items() if k != "list_fields"}
        return out

    def for_tally(self, *, skip_null: bool = False) -> Tuple[List, List]:
        complex_out: List[Tuple[Tuple[str], str]] = []
        simple_out: List[Tuple[str, str]] = []
        for k, v in self.__dict__.items():
            if k == 'list_fields':
                continue
            if skip_null and v is None:
                continue
            first_u = k.index('_')
            if v in self.list_fields:
                simple_out.update({k[:first_u]: l for l in v})
            else:
                complex_out[(k[:first_u], k[1+first_u:])] = v

        return simple_out, complex_out

    @classmethod
    def from_json(cls,
                  **kwargs):
        return cls(**kwargs)

@dataclass(frozen=True)
class MetaDataCollection:
    records: List[RecordMetaData]
    ids: List[str]

    def get_for_id(self,
                   id: str):
        ind = self.ids.index(id)

        return self.records[ind]

    def __getitem__(self, id) -> RecordMetaData:
        return self.get_for_id(id)

    def __contains__(self, id: str) -> bool:
        return id in self.ids

    def values_for_field(self,
                         field: str) -> List[str]:
        out = []
        for r in self.records:
            val = r.__getattribute__(field)
            if val is None:
                continue
            if isinstance(val, list):
                out.extend(val)
            assert isinstance(val, str)
            out.append(val)

        return [val]

    def unique_values_for_field(self,
                                field: str) -> List[str]:
        out = self.values_for_field(field)
        return sorted(list(set(out)))

    @staticmethod
    def run_api(id: str):
        raise NotImplementedError

    @classmethod
    def from_ids(cls,
                 ids: List[str],
                 batch_size: int = 64):
        records: List[RecordMetaData] = []
        for begin in range(0, len(ids), batch_size):
            id_batch = ids[begin:begin + batch_size]
            api_out = [MetaDataCollection.run_api(id) for id in id_batch]
            records.extend(
                [RecordMetaData.from_api(id, ao)
                 for id, ao in zip(api_out, api_out)]
            )

        return cls(records, ids)

    def to_dict(self) -> List[Dict]:
        return [v.to_dict() for v in self.records]

    @classmethod
    def from_jsonl(cls,
                  jsonl_file: str):
        records: List[RecordMetaData] = []
        ids: List[str] = []
        with open(jsonl_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                r = RecordMetaData.from_json(**(json.loads(line.strip())))
                records.append(r)
                ids.append(r.id)

        return cls(records, ids)