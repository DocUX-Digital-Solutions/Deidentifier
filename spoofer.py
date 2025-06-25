import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Callable, Set, Optional, Iterable

import numpy as np
from frozendict import FrozenOrderedDict

from date import Date, DateConverter
from ml_util.label_tokens import CompiledDocLabels, CompiledDocLabelSpoofer
from ml_util.multi import multi_cpu_map
from ml_util import docux_logger
logger = docux_logger.give_logger()


# most code copied from hide_in_plain_sight.py

def option_sort(options: Iterable[str]) -> List[str]:
    return sorted(list(options), key=lambda x: (-len(x), x))

def sets_for_tally(tally: Dict[str, int],
                   similarity_function: Callable,
                   min_count: int = 1) -> Tuple[List[Tuple], List[int]]:
    sim_sets: List[Set[str]] = []
    for f in option_sort(tally.keys()):
        matched = False
        for s in sim_sets:
            if any([similarity_function(f, s_f) for s_f in s]):
                s.add(f)
                matched = True
        if not matched:
            sim_sets.append({f})

    form_sets = []
    weights = []
    for s in sorted([frozenset(s) for s in sim_sets]):
        total = sum([tally[f] for f in s])
        if total >= min_count:
            form_sets.append(
                tuple(sorted(list(s), key=lambda x: (-len(x), x)))
            )
            weights.append(total)

    return form_sets, weights


@dataclass(frozen=True)
class TypeSpoofer:
    @staticmethod
    def create(*args):
        raise NotImplementedError

    def spoof_value(self,
                    old_values: Iterable[str],
                    count: int = 1) -> List[str]:
        raise NotImplementedError

    def spoof_values(self,
                     values: Dict[str, int]) -> Dict[Tuple: Iterable[str]]:
        return {form_set: self.spoof_value(form_set, count)
                for form_set, count in zip(*sets_for_tally(values, self.are_equal))}

    @staticmethod
    def are_equal(old, new) -> bool:
        raise NotImplementedError

    @staticmethod
    def basic_string_are_equal(old, new) -> bool:
        def clean(s):
            s = re.sub("\s+", "", s)
            return s.lower()

        return clean(old) == clean(new)


def get_types_and_counts(tally: Dict[str, int],
                         min_count: int =1) -> Tuple[List[str], List[int]]:
    l = []
    w = []
    for k, v in tally.items():
        if v >= min_count:
            l.append(k)
            w.append(int(v))

    return l, w


@dataclass(frozen=True)
class FieldSpoofer:
    form_sets: List[Tuple]
    weights: List[int]
    similarlity_function: Callable

    @classmethod
    def create(cls,
               tally: Dict[str, int],
               similarity_function: Callable,
               min_count: int = 1):
        form_sets, weights = sets_for_tally(tally, similarity_function, min_count)

        return cls(form_sets,
                   weights,
                   similarity_function)

    # DIGDI: Need to check uses..
    def give_random_item(self,
                         exclude_forms: Optional[Iterable[str]]) -> str:
        if exclude_forms is None:
            use_forms = self.form_sets
            use_weights = self.weights
        else:
            use_forms: List[Tuple] = []
            use_weights = []
            for form_set, weight in zip(self.form_sets, self.weights):
                if not any([self.similarlity_function(ef, form_set) for ef in exclude_forms]):
                    use_forms.append(form_set)
                    use_weights.append(weight)

        return random.choices(use_forms, weights=use_weights, k=1)[0][0]


@dataclass(frozen=True)
class VendorSpoofer(TypeSpoofer):
    spoofer: FieldSpoofer
    allow_equal_output: bool
    _max_tries: int

    @classmethod
    def create(cls,
               vendor_tally: Dict[str, int],
               min_count: int = 1,
               allow_equal_output: bool = False,
               _max_tries: int = 5):
        spoofer = FieldSpoofer.create(vendor_tally,
                                      VendorSpoofer.are_equal,
                                      min_count)
        if not allow_equal_output and len(spoofer.form_sets) <= 1:
            raise ValueError

        return cls(spoofer, allow_equal_output, _max_tries)

    @staticmethod
    def convert_vendor_to_short_without_numbers(vendor):
        return VendorSpoofer.convert_vendor_to_short(re.sub(r"[\d]", " ", vendor, flags=re.DOTALL))

    @staticmethod
    def convert_vendor_to_short_without_symbols(vendor):
        return VendorSpoofer.convert_vendor_to_short(re.sub(r"[^\w]", " ", vendor, flags=re.DOTALL))

    @staticmethod
    def convert_vendor_to_short_without_both(vendor):
        return VendorSpoofer.convert_vendor_to_short(
            re.sub(r"[^\w]|[\d]", " ", vendor, flags=re.DOTALL)
        )

    @staticmethod
    def convert_vendor_to_short(vendor):
        vendor_tokens = None
        vendor_tokens = vendor.split()

        if len(vendor_tokens) == 1:
            return vendor.title()
        return "".join(e[0] for e in vendor_tokens).upper()

    @staticmethod
    def are_equal(old_vendor, new_vendor):
        # If old_hospital is a plain word
        old_clean = set(
            re.sub(r"[^\w]", " ", old_vendor.lower(), flags=re.DOTALL).split()
        )
        new_clean = set(
            re.sub(r"[^\w]", " ", new_vendor.lower(), flags=re.DOTALL).split()
        )

        intersection_len = len(old_clean.intersection(new_clean))
        if (
            len(old_clean) > 0
            and len(new_clean) > 0
            and (
                intersection_len / len(old_clean) >= 0.66
                or intersection_len / len(new_clean) >= 0.66
            )
        ):
            return True

        # If old_hospital is an abbreviation
        old_clean = old_vendor.lower()
        new_clean1 = VendorSpoofer.convert_vendor_to_short(new_vendor).lower()
        new_clean2 = VendorSpoofer.convert_vendor_to_short_without_numbers(new_vendor).lower()
        new_clean3 = VendorSpoofer.convert_vendor_to_short_without_symbols(new_vendor).lower()
        new_clean4 = VendorSpoofer.convert_vendor_to_short_without_both(new_vendor).lower()

        if SequenceMatcher(None, old_clean, new_clean1).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, old_clean, new_clean2).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, old_clean, new_clean3).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, old_clean, new_clean4).ratio() >= 0.66:
            return True

        new_clean = new_vendor.lower()
        old_clean1 = VendorSpoofer.convert_vendor_to_short(old_vendor).lower()
        old_clean2 = VendorSpoofer.convert_vendor_to_short_without_numbers(old_vendor).lower()
        old_clean3 = VendorSpoofer.convert_vendor_to_short_without_symbols(old_vendor).lower()
        old_clean4 = VendorSpoofer.convert_vendor_to_short_without_both(old_vendor).lower()

        if SequenceMatcher(None, new_clean, old_clean1).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, new_clean, old_clean2).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, new_clean, old_clean3).ratio() >= 0.66:
            return True
        if SequenceMatcher(None, new_clean, old_clean4).ratio() >= 0.66:
            return True

        return False

    def spoof_value(self,
                    old_values: Iterable[str],
                    count: int = 1) -> List[str]:
        return count * [self.spoofer.give_random_item(old_values if not self.allow_equal_output else None)]


@dataclass(frozen=True)
class HospitalSpoofer(TypeSpoofer):
    hospital_spoofer: FieldSpoofer # Only handles finding a different hospital, not the format
    department_spoofer: FieldSpoofer
    format_weights: List[int]
    allow_equal_output: bool
    frequent_word_list: List[str]

    _formats = ("long_form", "short_form")

    @classmethod
    def create(cls,
               hospital_tally: Dict[str, int],
               department_tally: Dict[str, int],
               university_tally: Dict[str, int],
               hospital_format_tally: Dict[str, int],
               frequent_word_list: List[str],
               ratio_university: float = 0.6,
               allow_equal_output: bool = True):
        universe_size = 1000000
        assert len(hospital_tally) > 0
        hospital_scaling = sum(hospital_tally.values()) / float(universe_size)
        sources = {}
        if len(university_tally) > 0:
            hospital_scaling *= (1 - ratio_university)
            university_scaling = (ratio_university * sum(university_tally.values()) / float(universe_size))
            sources.update({k: university_scaling * v
                            for k, v in university_tally.items()})
        sources.update({k: hospital_scaling * v
                        for k, v in hospital_tally.items()})
        format_weights = [hospital_format_tally.get(f, 0) for f in HospitalSpoofer._formats]

        frequent_word_list.extend(["state", "college"])
        frequent_word_list = sorted(list(set(
            [w.strip().lower() for w in frequent_word_list])))

        return cls(FieldSpoofer.create(sources, HospitalSpoofer.are_equal),
                   FieldSpoofer.create(department_tally, TypeSpoofer.basic_string_are_equal),
                   format_weights,
                   allow_equal_output=allow_equal_output,
                   frequent_word_list=frequent_word_list,
                   )

    @staticmethod
    def are_equal(old_hospital, new_hospital) -> bool:
        # If old_hospital is a plain word
        old_clean = set(
            re.sub(r"[^\w]", " ", old_hospital.lower(), flags=re.DOTALL).split()
        )
        new_clean = set(
            re.sub(r"[^\w]", " ", new_hospital.lower(), flags=re.DOTALL).split()
        )

        intersection_len = len(old_clean.intersection(new_clean))
        if (
            len(old_clean) > 0
            and len(new_clean) > 0
            and (
                intersection_len / len(old_clean) >= 0.66
                or intersection_len / len(new_clean) >= 0.66
            )
        ):
            return True

        # If old_hospital is an abreviation
        old_clean = re.sub(r"[^\w]", "", old_hospital.lower(), flags=re.DOTALL)
        new_clean = re.sub(
            r"[^\w]",
            "",
            HospitalSpoofer.convert_hospital_to_short(new_hospital).lower(),
            flags=re.DOTALL,
        )

        if SequenceMatcher(None, old_clean, new_clean).ratio() >= 0.66:
            return True

        old_clean = re.sub(
            r"[^\w]",
            "",
            HospitalSpoofer.convert_hospital_to_short(old_hospital).lower(),
            flags=re.DOTALL,
        )
        new_clean = re.sub(r"[^\w]", "", new_hospital.lower(), flags=re.DOTALL)

        if SequenceMatcher(None, old_clean, new_clean).ratio() >= 0.66:
            return True

        return False

    def remove_hospital_tokens(self,
                               hospital: str):
        random_number = random.random()
        if random_number > 0.3:
            return hospital.title()

        hospital_tokens = hospital.split()
        if set(hospital_tokens).intersection(set(self.frequent_word_list)) == set(
            hospital_tokens
        ):
            return hospital.title()
        elif random_number > 0.15:
            for token in self.frequent_word_list:
                if token in hospital_tokens:
                    hospital_tokens.remove(token)
            return " ".join(hospital_tokens).title()
        else:
            tokens_in_common = list(
                set(hospital_tokens).intersection(set(self.frequent_word_list))
            )
            if len(tokens_in_common) == 0:
                return hospital.title()
            number_tokens_selected = random.randint(1, len(tokens_in_common))
            random.shuffle(tokens_in_common)
            tokens_in_common = tokens_in_common[:number_tokens_selected]
            for token in tokens_in_common:
                hospital_tokens.remove(token)

            return " ".join(hospital_tokens).title()


    def reverse_frequent_tokens(self,
                                hospital: str):
        if random.random() > 0.5:
            return hospital.title()

        hospital_split = hospital.split()
        start_index = 0
        current_tokens_are_frequent = False

        for i, token in enumerate(hospital_split):
            if token.lower() in self.frequent_word_list:
                if current_tokens_are_frequent:
                    continue
                else:
                    current_tokens_are_frequent = True
                    start_index = i

            else:
                if current_tokens_are_frequent:
                    current_tokens_are_frequent = False

        if not current_tokens_are_frequent:
            return hospital.title()

        hospital_split = (
            hospital_split[start_index:]
            + [random.choice(["of", "of", "at", ""])]
            + hospital_split[:start_index]
        )

        return " ".join(hospital_split).strip().replace("  ", " ").title()

    def add_department(self,
                       hospital: str,
                       parsed_hospital: List[str]):
        # hospital is the fake hospital
        # parsed_hospital is the true parsed hospital

        if random.random() > 0.2:
            return hospital.title()

        department = None
        number_iterations = 0

        while department is None or any(
            [
                SequenceMatcher(
                    None, parsed_hospital_token.lower(), department.lower()
                ).ratio()
                >= 0.7
                for parsed_hospital_token in parsed_hospital
            ]
        ):
            department = self.department_spoofer.give_random_item()
            number_iterations += 1
            if number_iterations > 15:
                raise Exception("problem")

        if random.random() > 0.5:
            return (
                department
                + random.choice([" of", " at", "", ",", " -"])
                + " "
                + hospital
            ).title()

        return (
            hospital + random.choice([" of", "", ",", " -"]) + " " + department
        ).title()

    def spoof_value(self,
                    old_hospital_strings: List[str],
                    count: int = 1) -> List[str]:
        parsed_hospital = old_hospital_strings[0].lower().split()

        hospital = self.hospital_spoofer.give_random_item(old_hospital_strings)

        formats = Counter(random.choices(self._formats, weights=self.format_weights, k=count))

        out = []
        for format, count in formats.items():
            # Ideally, this should be re-implimented to only require one pass of sampling.
            for _ in range(count):
                hospital = self.remove_hospital_tokens(hospital)
                hospital = self.reverse_frequent_tokens(hospital)
                if format == "long_form":
                    hospital = self.convert_hospital_to_long(hospital)
                else:
                    hospital = self.convert_hospital_to_short(hospital)
                out.append(
                    self.add_department(hospital, parsed_hospital)
                )
        random.shuffle(out)
        return out

    @staticmethod
    def parse_hospital_format(name: str) -> str:
        """Parse the format of a hospital which is a string
Returns its format as a string
        """
        if name.strip().count(" ") > 0:
            return "long_form"
        else:
            return "short_form"

    @staticmethod
    def count_hospital_format_frequencies(hospital_list):
        """Takes the list of all hospitals of the reports
Returns the number of hospitals per hospital format
        """
        frequencies_dict = {f: 0 for f in HospitalSpoofer._formats}

        for hospital in hospital_list:
            frequencies_dict[HospitalSpoofer.parse_hospital_format(hospital)] += 1

        # manual enforcement
        frequencies_dict = {
            "long_form": 3
            if frequencies_dict["long_form"] < frequencies_dict["short_form"]
            else 4,
            "short_form": 4
            if frequencies_dict["long_form"] < frequencies_dict["short_form"]
            else 3,
        }
        return frequencies_dict

    @staticmethod
    def convert_hospital_to_long(hospital: str):
        return hospital.title()

    @staticmethod
    def convert_hospital_to_short(hospital):
        hospital_tokens = None
        if random.random() > 0.2:
            hospital_tokens = re.sub(r"[^\w]", " ", hospital, flags=re.DOTALL).split()
        else:
            hospital_tokens = hospital.split()

        if (
            len(re.sub(r"[^\w]", " ", hospital, flags=re.DOTALL).split()) == 1
            or len(hospital.split()) == 1
        ):
            return hospital.title()

        if len("".join(e[0] for e in hospital_tokens).upper()) == 1:
            return hospital.title()

        return "".join(e[0] for e in hospital_tokens).upper()


@dataclass(frozen=True)
class UniqueSpoofer(TypeSpoofer):
    prev_spoofer: FieldSpoofer
    prob_new: float
    _num_tries: int

    @classmethod
    def create(cls,
               unique_tally: Dict[str, int],
               prob_new: float =0.3,
               num_tries: int = 5):
        return cls(FieldSpoofer.create(unique_tally, UniqueSpoofer.are_equal),
                   prob_new)

    @staticmethod
    def are_equal(old_unique, new_unique):
        # RETIRER LES CHARACTERES EVENTUELLEMENT
        # If old_hospital is a plain word
        old_clean = re.sub(r"[^\w]", "", old_unique.lower(), flags=re.DOTALL)
        new_clean = re.sub(r"[^\w]", "", new_unique.lower(), flags=re.DOTALL)

        if SequenceMatcher(None, old_clean, new_clean).ratio() >= 0.66:
            return True

        return False

    @staticmethod
    def generate_new_unique():
        unique_length = abs(int(np.random.normal(9, 2.5)))
        if unique_length <= 0:
            unique_length = 1

        number_letters = 0
        random_seed = random.random()
        if random_seed > 0.85:
            number_letters = random.randint(1, unique_length)
        elif random_seed > 0.7:
            number_letters = unique_length

        number_numbers = unique_length - number_letters

        letters = [random.choice(string.ascii_letters) for i in range(number_letters)]
        numbers = [random.randint(0, 9) for i in range(number_numbers)]

        unique = letters + numbers
        random.shuffle(unique)

        unique = "".join(map(str, unique))

        return unique

    @staticmethod
    def format_unique(unique: str):
        formatted_unique = unique
        random_seed = random.random()
        correct_prob = 1
        correct_prob_2 = 1

        if random_seed > 0.5:
            formatted_unique = formatted_unique.upper()
        elif random_seed > 0.35:
            formatted_unique = formatted_unique.lower()
            correct_prob_2 = 0.7
        else:
            correct_prob = 1.5

        if random.random() > 0.8:
            interval = abs(int(np.random.normal(3, 1)))
            if interval <= 0:
                interval = 1
            character = None
            seed_character = random.random()
            if seed_character > 0.3 * correct_prob:
                character = "-"
            elif seed_character > 0.2 * correct_prob:
                character = " "
            elif seed_character > 0.1 * correct_prob:
                character = "."
            else:
                character = "_"
            formatted_unique = character.join(
                re.findall(
                    r".{1," + str(interval) + "}", formatted_unique, flags=re.DOTALL
                )
            )

        if random.random() > 0.9 * correct_prob_2:
            formatted_unique = "#" + formatted_unique

        return formatted_unique

    def spoof_value(self,
                    old_unique_strings: Iterable[str],
                    count: int = 1) -> List[str]:
        unique = None
        if random.random() >= self.prob_new:
            unique = self.prev_spoofer.give_random_item(old_unique_strings)
        else:
            for _ in range(self._num_tries):
                unique = self.generate_new_unique()
                if not any([self.are_equal(old, unique) for old in old_unique_strings]):
                    break

        return [self.format_unique(unique) for _ in range(count)]


@dataclass(frozen=True)
class PhoneSpoofer(TypeSpoofer):
    spoofer: FieldSpoofer
    _max_tries: int
    prob_new: float

    @classmethod
    def create(cls,
               phone_tally: Dict[str, int],
               max_tries: int = 5,
               prob_new: float = 0.3):
        return cls(
            FieldSpoofer.create(phone_tally, PhoneSpoofer.are_equal),
            max_tries,
            prob_new)

    @staticmethod
    def are_equal(old_phone, new_phone) -> bool:
        # RETIRER LES CHARACTERES EVENTUELLEMENT
        # If old_hospital is a plain word
        old_clean = re.sub(r"[^\w]", "", old_phone.lower(), flags=re.DOTALL)
        new_clean = re.sub(r"[^\w]", "", new_phone.lower(), flags=re.DOTALL)

        if SequenceMatcher(None, old_clean, new_clean).ratio() >= 0.66:
            return True

        return False

    @staticmethod
    def format_phone(phone):
        formatted_phone = phone

        if random.random() > 0.05:
            formatted_phone = "(" + formatted_phone[:3] + ")" + formatted_phone[3:]

        if random.random() > 0.05:
            if formatted_phone[0] == "(":
                if random.random() > 0.05:
                    formatted_phone = formatted_phone[:5] + " " + formatted_phone[5:]
                else:
                    formatted_phone = formatted_phone[:5] + "-" + formatted_phone[5:]
            else:
                if random.random() > 0.05:
                    formatted_phone = formatted_phone[:3] + " " + formatted_phone[3:]
                else:
                    formatted_phone = formatted_phone[:3] + "-" + formatted_phone[3:]

        if random.random() > 0.05:
            formatted_phone = formatted_phone[:-4] + "-" + formatted_phone[-4:]

        if random.random() > 0.99:
            formatted_phone = "1" + formatted_phone

            if random.random() > 0.5:
                formatted_phone = formatted_phone[0] + " " + formatted_phone[1:]

            if random.random() > 0.5:
                formatted_phone = "+" + formatted_phone

        return formatted_phone

    def spoof_value(self,
                    old_phone_strings: List[str],
                    count: int = 1) -> List[str]:
        phone = None
        if random.random() >= self.prob_new:
            phone = self.spoofer.give_random_item(old_phone_strings)
        else:
            phone_length = 10

            for _ in range(self._max_tries):
                numbers = [random.randint(0, 9) for i in range(phone_length)]
                numbers[0] = random.randint(1, 9)

                phone = "".join(map(str, numbers))
                if not any([self.are_equal(old, phone) for old in old_phone_strings]):
                    break

        return [self.format_phone(phone) for _ in range(count)]


@dataclass(frozen=True)
class AgeSpoofer(TypeSpoofer):
    _num_tries: int
    min_age: int = 12
    high_age: int = 90
    max_age: int = 120

    @classmethod
    def create(cls,
               num_tries: int = 5):
        return cls(num_tries)

    @staticmethod
    def are_equal(old_age, new_age):
        old_age_without_letters = "".join([x for x in list(old_age) if x.isnumeric()])

        if int(old_age_without_letters if old_age_without_letters else 90) == int(
            new_age
        ):
            return True

        return False

    def spoof_value(self,
                    old_age_strings: List[str],
                    count: int = 1) -> List[str]:
        age = None
        for _ in range(self._num_tries):
            if random.random() < 0.05:
                age = random.randint(1 + self.high_age, self.max_age)
            else:
                age = random.randint(self.min_age, self.max_age)
            age = str(age)
            if not any([self.are_equal(old, age) for old in old_age_strings]):
                break

        return count * [age]


@dataclass(frozen=True)
class DateSpoofer(TypeSpoofer):
    min_date: Date
    max_date: Date
    date_format_types: List[str]
    date_format_counts: List[int]
    date_converter: DateConverter
    date_types: List[Date]
    date_counts: List[int]
    force_change: bool
    max_tries: int

    @classmethod
    def create(cls,
               min_date: Date,
               max_date: Date,
               date_format_tally: Dict[str, int],
               date_tally: Dict[Date, int],
               force_change: bool = False):
        date_converter = DateConverter()

        date_format_types = list(date_format_tally.keys())
        date_format_counts = [date_format_tally[k] for k in date_format_types]

        date_types = list(date_tally.keys())
        date_counts = [date_tally[k] for k in date_types]

        return cls(min_date,
                   max_date,
                   date_format_types,
                   date_format_counts,
                   date_converter,
                   date_types,
                   date_counts,
                   force_change=force_change,
                   max_tries=5)

    # compare to generate_date
    # Simplified  -- just sample date and format distributions; use constant for slash/hyphen
    def spoof_value(self,
                    parsed_date: Date,
                    count: int =1) -> List[str]:
        for _ in range(self.max_tries):
            new_date = random.choices(self.date_types, self.date_counts)
            if not self.force_change or new_date != parsed_date:
                break

        new_formats = Counter(random.choices(self.date_format_types, self.date_format_counts))
        out = []
        for format, n in new_formats.items():
            out += n * [self.date_converter.date_to_string(new_date, format)]
        random.shuffle(out)

        return out

    def spoof_values(self,
                     values: Dict[str, int]) -> Dict[Tuple: Iterable[str]]:
        date_forms: Dict[Date, Dict[str, int]] = defaultdict(dict)
        for raw, cnt in values.items():
            parsed_date, _ = self.date_converter.string_to_date_and_format(raw)
            date_forms[parsed_date][raw] += cnt

        out: Dict[Tuple: Iterable[str]] = {}
        for date, by_date in date_forms.items():
            total = sum(by_date.values())
            out[
                tuple(option_sort(by_date.keys()))
            ] = self.spoof_value(date, total)

        return out


@dataclass(frozen=True)
class NameSpoofer(TypeSpoofer):
    def spoof_value(self, given, middle, surname, credential, count: int) -> str:
        raise NotImplementedError

    @staticmethod
    def give_name_set(name: str) -> Set[str]:
        raise NotImplementedError

    def spoof_values(self,
                     values: Dict[str, int]) -> Dict[Tuple: Iterable[str]]:
        name_constellations = []
        for name, cnt in values.items():
            found = False
            s = self.give_name_set(name)
            for o in name_constellations:
                if len(o[0] & s) > 0:
                    o[0] &= s
                    o[1][name] += cnt
                    found = True
                    break
            if not found:
                name_constellations.append([s, {name: cnt}])

        out = {tuple(tally.keys()): self.spoof_value(string_set, sum(tally.values()))
               for string_set, tally in name_constellations}

        return out

@dataclass(frozen=True)
class HealthCareWorkerSpoofer(NameSpoofer):
    surname_spoofer: FieldSpoofer
    given_spoofer: FieldSpoofer
    credential_spoofer: FieldSpoofer
    _max_tries: int
    credentials = set(
        map(
            lambda x: x.lower(),
            [
                "MD",
                "Dr.",
                "Dr",
                "PA",
                "PA-C",
                "CRNP",
                "RN",
                "APNP",
                "APRN",
                "CNM",
                "CNP",
                "CRNA",
                "DNP",
                "LPN",
                "DO",
                "MBBS",
                "MB",
            ],
        )
    )

    name_format_weights: List[int]
    convert_name_to_string: Dict[str, Callable]

    _name_formats: Tuple[str] = ("NAME, NAME, CRE",
                                 "NAME, FIRSTNAME/NAME, CRE",
                                 "NAME NAME/CRE NAME/NAME CRE")

    @staticmethod
    def _create_credential_tally() -> Dict[str, int]:
        credential_total = 10000
        credential_tally: Dict[str, int] = {}
        for class_weight, members in (
                (0.85, ("MD", "Dr.")),
                (0.1, ["PA", "PA-C", "CRNP"]),
                (0.05, ["RN", "APNP", "APRN", "CNM", "CNP", "CRNA", "DNP", "LPN", "DO", "MBBS", "MB"])
        ):
            def give_out(ratio: float):
                return int(round((credential_total * class_weight * ratio) / float(len(members))))

            for credential in members:
                if credential == "Dr.":
                    credential_tally["Dr."] = give_out(0.8)
                    credential_tally["Dr"] = give_out(0.2)
                elif credential == "PA-C":
                    credential_tally["PA-C"] = give_out(0.95)
                    credential_tally["P.A.-C."] = give_out(0.05)
                else:
                    credential_tally[credential] = give_out(0.95)
                    credential_tally[
                        "".join([x + "." for x in credential])
                    ] = give_out(0.05)

            return credential_tally

    @classmethod
    def create(cls,
               surname_tally: Dict[str, int],
               given_tally: Dict[str, int],
               name_format_tally: Dict[str, int],
               min_surname_count: int = 1,
               min_given_count: int = 1,
               max_tries: int = 5):
        assert set(name_format_tally.keys()) in set(cls._name_formats)
        format_weights = [name_format_tally.get(f, 0)
                          for f in cls._name_formats]

        convert_name_to_string = \
            {"NAME, NAME, CRE": HealthCareWorkerSpoofer.convert_name_to_format_1,
             "NAME, FIRSTNAME/NAME, CRE": HealthCareWorkerSpoofer.convert_name_to_format_2,
             "NAME NAME/CRE NAME/NAME CRE": HealthCareWorkerSpoofer.convert_name_to_format_3}

        return cls(FieldSpoofer.create(surname_tally, HealthCareWorkerSpoofer.are_equal_field, min_surname_count),
                   FieldSpoofer.create(given_tally, HealthCareWorkerSpoofer.are_equal_field, min_given_count),
                   FieldSpoofer.create(HealthCareWorkerSpoofer._create_credential_tally(),
                                       HealthCareWorkerSpoofer.are_equal_field),
                   name_format_weights=format_weights,
                   convert_name_to_string=convert_name_to_string,
                   _max_tries=max_tries)

    @staticmethod
    def give_name_set(name: str) -> Set[str]:
            return (set(re.sub(r"[^\w]", " ", name, flags=re.DOTALL).lower().split())
                    - HealthCareWorkerSpoofer.credentials)

    @staticmethod
    def are_equal(old_name, new_name) -> bool:
        return len(HealthCareWorkerSpoofer.give_name_set(old_name)
                   & (HealthCareWorkerSpoofer.give_name_set(new_name))) > 0

    @staticmethod
    def are_equal_field(old: Set[str],
                        new: str) -> bool:
        return bool(new.lower() in old)

    @staticmethod
    def parse_name_format(name):
        """Parse the format of a name which is a string
        Returns its format as a string
         """
        if name.count(",") == 2:
            return "NAME, NAME, CRE"
        elif name.count(",") == 1:
            return "NAME, FIRSTNAME/NAME, CRE"
        else:
            return "NAME NAME/CRE NAME/NAME CRE"

    @staticmethod
    def count_name_format_frequencies(name_list) -> Dict[str, int]:
        """Takes the list of all names of the reports
Returns the number of names per name format
        """
        frequencies_dict = \
            {n: 0
             for n in HealthCareWorkerSpoofer._name_formats}

        for name in name_list:
            frequencies_dict[HealthCareWorkerSpoofer.parse_name_format(name)] += 1

        return frequencies_dict

    # Copied wholesale -- losts of "magic numbers" here...

    @staticmethod
    def convert_name_to_format_1(firstname, middlename, lastname, credential):
        random_number = random.random()
        if random_number > 0.9:
            firstname += " " + middlename[0]
            if random.random() > 0.8:
                firstname += "."
        elif random_number > 0.8:
            firstname += " " + middlename

        if credential[0:2].lower() == "dr" and random.random() > 0.1:
            return "{} {}, {}".format(credential, lastname, firstname)

        if random.random() > 0.5:
            return "{}, {}, {}".format(firstname, lastname, credential)
        else:
            return "{}, {}, {}".format(lastname, firstname, credential)

    @staticmethod
    def convert_name_to_format_2(firstname, middlename, lastname, credential):
        random_number = random.random()
        if random_number > 0.9:
            firstname += " " + middlename[0]
            if random.random() > 0.6:
                firstname += "."
        elif random_number > 0.8:
            firstname += " " + middlename

        prob_list = [0.875, 0.75, 0.5, 0.25, 0.125]
        if credential[0:2].lower() == "dr":
            prob_list = [0.6, 0.2, 0.15, 0.1, 0.05]

        random_number = random.random()
        if random_number > prob_list[0]:
            return "{}, {}".format(firstname, lastname)
        elif random_number > prob_list[1]:
            return "{}, {}".format(lastname, firstname)
        elif random_number > prob_list[2]:
            return "{} {}, {}".format(firstname, lastname, credential)
        elif random_number > prob_list[3]:
            return "{} {}, {}".format(lastname, firstname, credential)
        elif random_number > prob_list[4]:
            return "{}, {}".format(firstname, credential)
        else:
            return "{}, {}".format(lastname, credential)

    @staticmethod
    def convert_name_to_format_3(firstname, middlename, lastname, credential):
        random_number = random.random()
        if random_number > 0.9:
            firstname += " " + middlename[0]
            if random.random() > 0.8:
                firstname += "."
        elif random_number > 0.8:
            firstname += " " + middlename

        random_number = random.random()
        prob_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        if credential[0:2].lower() == "dr":
            prob_list = [0.93, 0.86, 0.84, 0.82, 0.58, 0.34, 0.32, 0.3, 0.14]
        if random_number > prob_list[0]:
            return "{} {}".format(firstname, lastname)
        elif random_number > prob_list[1]:
            return "{} {}".format(lastname, firstname)
        elif random_number > prob_list[2]:
            return "{} {} {}".format(firstname, lastname, credential)
        elif random_number > prob_list[3]:
            return "{} {} {}".format(lastname, firstname, credential)
        elif random_number > prob_list[4]:
            return "{} {} {}".format(credential, lastname, firstname)
        elif random_number > prob_list[5]:
            return "{} {} {}".format(credential, firstname, lastname)
        elif random_number > prob_list[6]:
            return "{} {}".format(firstname, credential)
        elif random_number > prob_list[7]:
            return "{} {}".format(lastname, credential)
        elif random_number > prob_list[8]:
            return "{} {}".format(credential, lastname)
        else:
            return "{} {}".format(credential, firstname)

    def generate_random_name(self, name_strings: Set[str]) -> Tuple[str, str, str, str]:
        return (self.given_spoofer.give_random_item(name_strings),
                self.given_spoofer.give_random_item(name_strings),
                self.surname_spoofer.give_random_item(name_strings),
                self.credential_spoofer.give_random_item(None))

    def spoof_value(self,
                    name_strings: Set[str],
                    count: int = 1) -> List[str]:
        new_name_fields = self.generate_random_name(name_strings)
        use_formats = Counter(random.choices(self._name_formats, weights=self.name_format_weights, k=count))
        out = []
        for format, count in use_formats.items():
            out += count * [self.convert_name_to_string[format](*new_name_fields)]

        random.shuffle(out)
        return out


@dataclass(frozen=True)
class PatientSpoofer(NameSpoofer):
    surname_spoofer: FieldSpoofer
    given_spoofer: FieldSpoofer
    honorific_spoofer: FieldSpoofer
    _max_tries: int

    _honorifics = set([h.lower
                       for h in ("Mr", "Mrs", "Miss", "Ms", "Mx", "Sir", "Dr", "Lady", "Lord")
                       ])

    @staticmethod
    def _create_honorific_tally() -> Dict[str, int]:
        honorific_total = 10000
        honorific_tally: Dict[str, int] = {}
        for class_weight, members in (
                (0.99 * 0.5, ("Mr")),
                (0.99 * 0.5, ("Mrs", "Miss", "Ms")),
                (0.01 - 0.005, ("Mx", "Sir", "Dr")),
                (0.005, ("Lady", "Lord"))
        ):
            def give_out(ratio: float):
                return int(round((honorific_total * class_weight * ratio) / float(len(members))))

            for honorific in members:
                honorific_tally[honorific] = give_out(0.7)
                honorific_tally[honorific + "."] = give_out(0.7)

            return honorific_tally

    @classmethod
    def create(cls,
               surname_tally: Dict[str, int],
               given_tally: Dict[str, int],
               min_surname_count: int = 1,
               min_given_count: int = 1,
               max_tries: int = 5):
        return cls(FieldSpoofer.create(surname_tally, PatientSpoofer.are_equal, min_surname_count),
                   FieldSpoofer.create(given_tally, PatientSpoofer.are_equal, min_given_count),
                   FieldSpoofer.create(PatientSpoofer._create_honorific_tally(), PatientSpoofer.are_equal),
                   _max_tries=max_tries)

    @staticmethod
    def convert_patient_to_string(firstname, middlename, lastname, honorific):
        random_number = random.random()

        if random_number > 0.95:
            firstname += " " + middlename[0]
            if random.random() > 0.6:
                firstname += "."
        elif random_number > 0.9:
            firstname += " " + middlename

        random_number = random.random()

        if random_number > 0.45:
            return "{} {}".format(firstname, lastname)
        elif random_number > 0.35:
            return "{} {}".format(lastname, firstname)
        elif random_number > 0.25:
            return "{} {} {}".format(honorific, lastname, firstname)
        elif random_number > 0.15:
            return "{} {} {}".format(honorific, firstname, lastname)
        elif random_number > 0.05:
            return "{} {}".format(honorific, lastname)
        else:
            return "{} {}".format(honorific, firstname)

    @staticmethod
    def give_name_set(name: str) -> Set[str]:
        return set([n for n in
                    re.sub(r"[^\w]", " ", name, flags=re.DOTALL).lower().split()
                    if n.rstrip() not in PatientSpoofer._honorifics])


    def spoof_value(self, given, middle, surname, credential, count: int) -> List[str]:
        new_name_fields = (self.given_spoofer.give_random_item(given),
                           self.given_spoofer.give_random_item(middle),
                           self.surname_spoofer.give_random_item(surname),
                           self.honorific_spoofer.give_random_item(credential))

        return [self.convert_patient_to_string(*new_name_fields) for _ in range(count)]


@dataclass(frozen=True)
class Spoofer:
    # dict_generate_phi: Dict
    # dict_generate_constraint: Dict
    type_spoofers: Dict[str, TypeSpoofer]

    @classmethod
    def create(cls,
               value_tallies: Dict[str, Dict[str, int]],
               date_tally: Dict[Date, int],
               date_format_tally: Dict[str, int],
               hcw_surname_tally: Dict[str, int],
               hcw_given_tally: Dict[str, int],
               hcw_name_format_tally: Dict[str, int],
               patient_surname_tally: Dict[str, int],
               patient_given_tally: Dict[str, int],

               hospital_tally: Dict[str, int],
               department_tally: Dict[str, int],
               university_tally: Dict[str, int],
               hospital_format_tally: Dict[str, int],
               frequent_word_list: List[str],

               unique_tally: Dict[str, int],
               phone_tally: Dict[str, int],
               ):
        '''
        Need:
        dict_generate_phi,
        dict_generate_constraint,
        '''

        all_dates = sorted(list(date_tally.keys()))

        # Need to indicate field names -- make into a dictionary??

        type_spoofers = {
            "VENDOR": VendorSpoofer.create(value_tallies['VENDOR']),
            "DATES": DateSpoofer.create(min_date=all_dates[0],
                                        max_date=all_dates[-1],
                                        date_tally=date_tally,
                                        date_format_tally=date_format_tally),
            "HCW": HealthCareWorkerSpoofer.create(hcw_surname_tally,
                                                  hcw_given_tally,
                                                  hcw_name_format_tally),
            "PATIENT": PatientSpoofer.create(patient_surname_tally,
                                             patient_given_tally),
            "HOSPITAL": HospitalSpoofer.create(hospital_tally,
                                               department_tally,
                                               university_tally,
                                               hospital_format_tally,
                                               frequent_word_list),
            "UNIQUE": UniqueSpoofer.create(unique_tally),
            "PHONE": PhoneSpoofer.create(phone_tally),
            "AGE": AgeSpoofer.create(),
        }

        return cls(type_spoofers)

    ## Based on: generate_deidentified_report
    def process_report(self,
                       report: str,
                       labels: CompiledDocLabels,
                       show_mappings: bool = False,
                       ) -> Tuple[str, FrozenOrderedDict]:
        spoofer = CompiledDocLabelSpoofer(report, labels)
        # Add mappings...
        for label, tallies in labels.as_tally().items():
            self.type_spoofers[label].spoof_values(tallies)

        return spoofer.render_new(show_mappings)

    def run(self,
            reports: List[str],
            label_sets: List[CompiledDocLabels],
            ) -> Tuple[List[str], List[FrozenOrderedDict]]:
        assert len(reports) == len(label_sets)
        raw_out = multi_cpu_map(self.process_report, (reports, label_sets))

        return [p[0] for p in raw_out], [p[1] for p in raw_out]
