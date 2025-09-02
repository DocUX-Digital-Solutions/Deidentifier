from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Set, Dict, List, Tuple

from lib.metadata import MetaDataCollection, RecordMetaData
from ml_util.data import TallyCluster, BaseTally, TallyClusterBuilder, BaseTallyBuilder
from ml_util.label_tokens import CompiledDocLabelTally, CompiledDocLabels
from ml_util.list_file import load_list_file


class HCW_NewCluster(TallyCluster):
    @property
    def given(self) -> BaseTally:
        return self.tallies['given']

    @property
    def surname(self) -> BaseTally:
        return self.tallies['surname']

    @property
    def middle(self) -> BaseTally:
        return self.tallies['middle']

    @property
    def name_format(self) -> BaseTally:
        return self.tallies['name_format']


class HCW_NewClusterBuilder(TallyClusterBuilder, HCW_NewCluster):
    def __init__(self,
                 **kwargs):
        super().__init__(required_fields=DEID_ClustersBuilder.get_local_properies(self), **kwargs)

    def render(self) -> HCW_NewCluster:
        return HCW_NewCluster(self.pre_render())


class Patient_NewCluster(TallyCluster):
    @property
    def given(self) -> BaseTally:
        return self.tallies['given']

    @property
    def preferred(self) -> BaseTally:
        return self.tallies['preferred']

    @property
    def middle(self) -> BaseTally:
        return self.tallies['middle']

    @property
    def surname(self) -> BaseTally:
        return self.tallies['surname']

    @property
    def sex(self) -> BaseTally:
        return self.tallies['sex']

    @property
    def age(self) -> BaseTally:
        return self.tallies['age']

    @property
    def name_format(self) -> BaseTally:
        return self.tallies['name_format']


class Patient_NewClusterBuilder(TallyClusterBuilder, Patient_NewCluster):
    def __init__(self,
                 **kwargs):
        super().__init__(required_fields=DEID_ClustersBuilder.get_local_properies(self), **kwargs)

    def render(self) -> Patient_NewCluster:
        return Patient_NewCluster(self.pre_render())


class Hospital_NewCluster(TallyCluster):
    @property
    def hospital(self) -> BaseTally:
        return self.tallies['hospital']

    @property
    def department(self) -> BaseTally:
        return self.tallies['department']

    @property
    def university(self) -> BaseTally:
        return self.tallies['uninversity']


class Hospital_NewClusterBuilder(TallyClusterBuilder, Hospital_NewCluster):
    def __init__(self,
                 **kwargs):
        super().__init__(required_fields=DEID_ClustersBuilder.get_local_properies(self), **kwargs)

    def render(self) -> Hospital_NewCluster:
        return Hospital_NewCluster(self.pre_render())


@dataclass(frozen=True)
class ParsedName:
    given: str = None
    middle: str = None
    surname: str = None
    honorific: str = None

    @property
    def credential(self) -> Optional[str]:
        return self.honorific

    @property
    def fields(self) -> Set[str]:
        raw = [n for n in (self.given, self.middle, self.surname) if n is not None]
        if len(raw) < 1:
            raise ValueError

        return set(raw)

# Should this be nested?
HCW_NameFormats: Tuple[str] = ("NAME, NAME, CRE",
                               "NAME, FIRSTNAME/NAME, CRE",
                               "NAME NAME/CRE NAME/NAME CRE")


def parse_name_format(name) -> str:
    """Parse the format of a name which is a string
    Returns its format as a string
     """
    if name.count(",") == 2:
        return "NAME, NAME, CRE"
    elif name.count(",") == 1:
        return "NAME, FIRSTNAME/NAME, CRE"
    else:
        return "NAME NAME/CRE NAME/NAME CRE"


@dataclass(frozen=True)
class HCW_Cluster:
    given: BaseTally
    surname: BaseTally
    name_format: BaseTally

    @classmethod
    def create(cls,
               inputs: CompiledDocLabelTally,
               *,
               given_name_file: str = None,
               surname_file: str = None,
               ):
        given_names = [] if given_name_file is None else load_list_file(given_name_file)
        surnames = [] if surname_file is None else load_list_file(surname_file)

        format_tally = {n: 0
                        for n in HCW_NameFormats}

        observed: Dict[ParsedName: Dict[str, int]] = defaultdict(dict)
        for raw_name, count in inputs['HCW'].items():
            name_format = parse_name_format(raw_name)
            format_tally[name_format] += count

        # DIGDI

@dataclass(frozen=True)
class DEID_Clusters:
    hcw: HCW_NewCluster
    patient: Patient_NewCluster
    hospital: Hospital_NewCluster
    unique: BaseTally
    phone: BaseTally
    #
    # hcw: HCW_Cluster  # health care worker
    # vendor: BaseTally
    # unique: BaseTally
    # phone: BaseTally

    def __getattr__(self, item):
        found = self.fields.get(item, None)
        if found is None:
            raise AttributeError
        return found

    @classmethod
    def create(cls,
               comp_doc_labels: List[CompiledDocLabels],
               mdc: MetaDataCollection,
               vendors: List[str],
               # *,
               # unique_json_file: str = None,
               # phone_json_file: str = None,
               # hcw_json_file: str = None
               ):
        mdc_clusters = DEID_ClustersBuilder()
        mdc_clusters.addMetaCollection(mdc)

        base: CompiledDocLabelTally = CompiledDocLabels.combined_tallies(comp_doc_labels)

        return cls(
            hcw_cluster,
            overload_or_key(vendor_json_file, 'VENDOR', str),
            overload_or_key(unique_json_file, 'UNIQUE', str),
            overload_or_key(phone_json_file, 'PHONE', str),
        )


class DEID_ClustersBuilder:
    def __init__(self):
        self.hcw = HCW_NewClusterBuilder()
        self.patient = Patient_NewClusterBuilder()
        self.hospital = Hospital_NewClusterBuilder()
        self.unique = BaseTallyBuilder()
        self.phone = BaseTallyBuilder()

        self.get_attrib = lambda x: self.__getattribute__(x)

    def addRecordMeta(self,
                      record: RecordMetaData):
        simple, complex = record.for_tally(skip_null=True)

        for f, v in simple:
            self.get_attrib(f).add_item(v)

        for (f0, f1), v in complex:
            self.get_attrib(f0).__getattribute__(f1).add_item(v)


    def addMetaCollection(self,
                          collection: MetaDataCollection):
        for record in collection.records:
            self.addRecordMeta(record)

    def render(self):
        return DEID_Clusters(self.hcw.render(),
                             self.patient.render(),
                             self.hospital.render(),
                             self.unique.render(),
                             self.phone.render())
