import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Set
import numpy as np

from ml_util.label_tokens import TokenClassificationOutputEditor
from ml_util.classes import TokenClassificationOutput

'''
* Working with labels for ages, hospitals and vendors.
* Impose well-formedness of labels.
'''


@dataclass(frozen=True)
class RuleBasedModel:
    @classmethod
    def create(cls):
        raise NotImplementedError

    def run(self,
            report: str) -> List[TokenClassificationOutput]:
        raise NotImplementedError


@dataclass(frozen=True)
class RuleBasedAgeDetector(RuleBasedModel):
    ages: Set[str]
    age_lengths: Set[int]
    prefix_to_exclude: Set[str]
    prefix_to_exclude_lengths: Set[int]
    suffix_to_exclude: Set[str]
    suffix_to_exclude_lengths: Set[int]

    def run(self,
            report: str) -> List[TokenClassificationOutput]:
        i = 0
        predictions: List[TokenClassificationOutput] = []

        while i < len(report):
            for length in self.age_lengths:
                if i + length <= len(report) and report[i: i + length].lower() in ages:

                    if report[i: i + length].isnumeric() and i + length < len(report):
                        if report[i + length].isnumeric():
                            continue

                        need_to_continue = False
                        for length_2 in self.suffix_to_exclude_lengths:
                            if (
                                    i + length + length_2 <= len(report)
                                    and report[i + length: i + length + length_2].lower()
                                    in self.suffix_to_exclude
                            ):
                                need_to_continue = True
                                break
                        if need_to_continue:
                            continue

                    if report[i: i + length].isnumeric() and i > 0:
                        if report[i - 1].isnumeric():
                            continue

                        need_to_continue = False
                        for length_2 in self.prefix_to_exclude_lengths:
                            if (
                                    i - length_2 >= 0
                                    and report[i - length_2: i].lower()
                                    in self.prefix_to_exclude
                            ):
                                need_to_continue = True
                                break
                        if need_to_continue:
                            continue

                    predictions.append(
                        TokenClassificationOutput(label="AGE",
                                                  score=0.9,
                                                  start_char=i,
                                                  exclusive_end_char=i+length)
                    )
                    i += length
                    break
            else:
                i += 1

        return predictions

    @classmethod
    def create(cls):
        ages = set(
            [
                "90",
                "91",
                "92",
                "93",
                "94",
                "95",
                "96",
                "97",
                "98",
                "99",
                "100",
                "101",
                "102",
                "103",
                "104",
                "105",
                "106",
                "107",
                "108",
                "109",
                "110",
                "111",
                "112",
                "113",
                "114",
                "115",
                "116",
                "117",
                "118",
                "119",
                "120",
                "121",
                "122",
                "123",
                "124",
                "125",
                "126",
                "127",
                "128",
                "129",
                "130",
                "ninety",
                "ninety-one",
                "ninety-two",
                "ninety-three",
                "ninety-four",
                "ninety-five",
                "ninety-six",
                "ninety-seven",
                "ninety-eight",
                "ninety-nine",
                "ninety",
                "ninety one",
                "ninety two",
                "ninety three",
                "ninety four",
                "ninety five",
                "ninety six",
                "ninety seven",
                "ninety eight",
                "ninety nine",
                "ninety",
                "ninetyone",
                "ninetytwo",
                "ninetythree",
                "ninetyfour",
                "ninetyfive",
                "ninetysix",
                "ninetyseven",
                "ninetyeight",
                "ninetynine",
            ]
        )

        ages_lengths = set(map(len, ages))

        suffix_to_exclude = set(
            map(
                lambda x: x.lower(),
                [
                    " ml",
                    "  ml",
                    "/",
                    " mg",
                    "mg",
                    "  mg",
                    "ml",
                    "mg",
                    " /",
                    "-ML",
                    " mL",
                    "mL",
                    "  mL",
                    " ml",
                    "  ml",
                    "ml",
                    ".0",
                    ".1",
                    ".2",
                    ".3",
                    ".4",
                    ".5",
                    ".6",
                    ".7",
                    ".8",
                    ".9",
                    " cc",
                    "  mL",
                    "th",
                    " beats",
                    "m",
                    " cm",
                    "  cm",
                    "cm",
                    " mm",
                    "  mm",
                    "mm",
                    " mcg",
                    "  mcg",
                    "mcg",
                    "  minutes",
                    " minutes",
                    "minutes",
                    " g/m",
                    "g/m",
                    "  g/m",
                    " bpm",
                    "bpm",
                    "  bpm",
                    "  cc",
                    "cc",
                    " cc",
                    "%",
                    "grams",
                    " grams",
                    "  grams",
                ],
            )
        )
        prefix_to_exclude = set(
            map(
                lambda x: x.lower(),
                ["0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "image "],
            )
        )

        suffix_to_exclude_lengths = set(map(len, suffix_to_exclude))
        prefix_to_exclude_lengths = set(map(len, prefix_to_exclude))

        return cls(ages, ages_lengths,
                   prefix_to_exclude, prefix_to_exclude_lengths,
                   suffix_to_exclude, suffix_to_exclude_lengths)


@dataclass(frozen=True)
class RuleBasedSimpleModel(RuleBasedModel):
    label: str
    targets: Set[str]
    target_lengths: Set[int]

    @classmethod
    def create(cls,
               label: str,
               targets: List[str],
               add_at: bool = False):
        targets = set([x.lower() for x in targets])
        if add_at and "@" not in targets:
            targets.add("@")
        target_lengths = set(map(len, targets))

        return cls(label, targets, target_lengths)

    def run(self,
            report: str) -> List[TokenClassificationOutput]:
        i = 0
        predictions: List[TokenClassificationOutput] = []

        while i < len(report):
            for length in self.target_lengths:
                if (
                    i + length <= len(report)
                    and report[i : i + length].lower() in self.targets
                ):
                    predictions.append(
                        TokenClassificationOutput(label=self.label,
                                                  score=0.9,
                                                  start_char=i,
                                                  exclusive_end_char=i + length)
                    )
                    i += length
                    break
            else:
                i += 1

        return predictions


@dataclass(frozen=True)
class LabelPostprocessingModel:
    rule_based_models: List[RuleBasedModel]
    characters_that_can_be_skipped: List[str]

    def run(self,
            reports: List[str],
            predictions: List[List[TokenClassificationOutput]]) -> List[List[TokenClassificationOutput]]:
        assert len(reports) == len(predictions)
        for i in range(len(reports)):
            for rule_based_model in self.rule_based_models:
                prediction_rule_based = rule_based_model.run(reports[i])
                predictions[i] = self.merge_predictions(
                    predictions[i], prediction_rule_based, reports[i]
                )

        for i in range(len(reports)):
            predictions[i] = self.propagate_predictions_to_letters_around(
                predictions[i], reports[i]
            )
        self.check_ordering(predictions)

        for i in range(len(reports)):
            predictions[i] = self.fuse_continuous_predictions(predictions[i], reports[i])
        self.check_ordering(predictions)

        for i in range(len(reports)):
            predictions[i] = self.fuse_neighbor_predictions_from_the_same_class(predictions[i], reports[i])
        self.check_ordering(predictions)

        # ignore model_labels_to_hips_labels
        # do *not* insert labels into the text.

        return predictions

    def fuse_continuous_predictions(self,
                                    prediction_list: List[TokenClassificationOutput],
                                    report: str):
        # if a continuous span of characters has different labels, we fuse into one span with one label
        index = 0
        bit_map = self.generate_bit_map_prediction(prediction_list, report)

        while (
            index < len(prediction_list) - 1
        ):  # not a problem in python if prediction_list is mutated during the execution
            # of the inner loop

            if prediction_list[index].exclusive_end_char < prediction_list[index + 1].start_char:
                index += 1

            elif prediction_list[index].exclusive_end_char > prediction_list[index + 1].start_char:
                raise Exception("overlapping")

            else:
                assert (
                    prediction_list[index].exclusive_end_char == prediction_list[index + 1].start_char
                )
                start_index = index
                label_to_prob = defaultdict(int)
                label_to_prob[prediction_list[index].label] += prediction_list[
                    index
                ].score * (
                        prediction_list[index].exclusive_end_char - prediction_list[index].start_char
                    )

                while (
                    index + 1 < len(prediction_list)
                    and prediction_list[index].exclusive_end_char
                    == prediction_list[index + 1].start_char
                ):
                    index += 1
                    label_to_prob[prediction_list[index].label] += prediction_list[
                        index
                    ].score * (
                            prediction_list[index].exclusive_end_char - prediction_list[index].start_char
                        )

                end_index = index

                label_to_assign = max(label_to_prob, key=lambda x: label_to_prob[x])
                prob_to_assign = max(label_to_prob.values())

                temp = TokenClassificationOutputEditor(prediction_list[start_index])
                temp.label = label_to_assign
                temp.score = prob_to_assign
                temp.exclusive_end_char = prediction_list[end_index].exclusive_end_char
                prediction_list[start_index] = temp.render()

                for _ in range(start_index + 1, end_index + 1):
                    prediction_list.pop(start_index + 1)

                index = start_index + 1

        assert not (
            bit_map != self.generate_bit_map_prediction(prediction_list, report)
        ).sum()

        return prediction_list

    def check_ordering(self,
                       predictions: List[List[TokenClassificationOutput]]):
        # Check that the predictions are correctly ordered, and no overlapping predictions
        for prediction in predictions:
            for i in range(len(prediction) - 1):
                assert prediction[i].start_char < prediction[i].exclusive_end_char
                assert prediction[i].exclusive_end_char <= prediction[i + 1].start_char
            if len(prediction):
                assert prediction[-1].start_char < prediction[-1].exclusive_end_char

    def propagate_predictions_to_letters_around(self,
                                                prediction_list: List[TokenClassificationOutput],
                                                report: str):
        prediction_index = 0
        bit_map = self.generate_bit_map_prediction(prediction_list, report)

        while prediction_index < len(prediction_list):
            # We must propagate on the left and on the right

            # the left limit is included
            left_limit_report_index = (
                prediction_list[prediction_index - 1].exclusive_end_char
                if prediction_index > 0
                else 0
            )
            # the right limit is not included
            right_limit_report_index = (
                prediction_list[prediction_index + 1].start_char
                if prediction_index < len(prediction_list) - 1
                else len(report)
            )

            start_index = prediction_list[prediction_index].start_char
            end_index = prediction_list[prediction_index].exclusive_end_char

            while (
                start_index > left_limit_report_index
                and report[start_index - 1].isalnum()
            ):
                start_index -= 1

            while end_index < right_limit_report_index and report[end_index].isalnum():
                end_index += 1

            temp = TokenClassificationOutputEditor(prediction_list[prediction_index])
            temp.start_char = start_index
            temp.exclusive_end_char = end_index
            prediction_list[prediction_index] = temp.render()
            prediction_index += 1

        assert not (
            bit_map
            & np.logical_not(self.generate_bit_map_prediction(prediction_list, report))
        ).sum()

        return prediction_list

    def merge_predictions(self,
                          prediction_list: List[TokenClassificationOutput],
                          prediction_rule_based_list: List[TokenClassificationOutput],
                          report: str):
        prediction_list_save = copy.deepcopy(prediction_list)
        bit_map = self.generate_bit_map_prediction(prediction_list, report)

        # per report
        for i in range(len(prediction_list) - 1):
            assert prediction_list[i].start_char < prediction_list[i].exclusive_end_char
            assert prediction_list[i].exclusive_end_char <= prediction_list[i + 1].start_char

        if len(prediction_list):
            assert prediction_list[-1].start_char < prediction_list[-1].exclusive_end_char

        for i in range(len(prediction_rule_based_list) - 1):
            assert (
                prediction_rule_based_list[i].start_char
                < prediction_rule_based_list[i].exclusive_end_char
            )
            assert (
                prediction_rule_based_list[i].start_char
                <= prediction_rule_based_list[i + 1].start_char
            )

        self.check_ordering([prediction_list])

        if len(prediction_rule_based_list):
            assert (
                prediction_rule_based_list[-1].start_char
                < prediction_rule_based_list[-1].exclusive_end_char
            )

        prediction_list_index = 0
        prediction_rule_based_list_index = 0

        while prediction_rule_based_list_index < len(
            prediction_rule_based_list
        ) and prediction_list_index < len(prediction_list):

            if (
                prediction_list[prediction_list_index].exclusive_end_char
                <= prediction_rule_based_list[prediction_rule_based_list_index].start_char
            ):
                # normal_end <= rule_based_start
                prediction_list_index += 1

            else:
                # normal_end > rule_based_start

                if (
                    prediction_list[prediction_list_index].start_char
                    >= prediction_rule_based_list[prediction_rule_based_list_index].exclusive_end_char
                ):
                    # normal_start >= rule_based_end
                    prediction_list.insert(
                        prediction_list_index,
                        prediction_rule_based_list[prediction_rule_based_list_index],
                    )
                    prediction_list_index += 1
                    prediction_rule_based_list_index += 1

                elif (
                    prediction_list[prediction_list_index].start_char
                    <= prediction_rule_based_list[prediction_rule_based_list_index].start_char
                    and prediction_list[prediction_list_index].exclusive_end_char
                    >= prediction_rule_based_list[prediction_rule_based_list_index].exclusive_end_char
                ):
                    # normal_start <= rule_based_start and normal_end >= rule_based_end
                    prediction_rule_based_list_index += 1

                # elif prediction_list[prediction_list_index]['start'] >= \
                # prediction_rule_based_list[prediction_rule_based_list_index]['start'] and\
                # prediction_list[prediction_list_index]['end'] <= \
                # prediction_rule_based_list[prediction_rule_based_list_index]['end']:
                #    #normal_start >= rule_based_start and normal_end <= rule_based_end
                else:
                    if (
                        prediction_list[prediction_list_index].start_char
                        > prediction_rule_based_list[prediction_rule_based_list_index].start_char
                    ):
                        temp = TokenClassificationOutputEditor(
                            prediction_rule_based_list[prediction_rule_based_list_index])
                        temp.exclusive_end_char = prediction_list[prediction_list_index].start_char
                        prediction_list.insert(prediction_list_index, temp.render())
                        prediction_list_index += 1

                    if (
                        prediction_list[prediction_list_index].exclusive_end_char
                        < prediction_rule_based_list[prediction_rule_based_list_index].exclusive_end_char
                    ):
                        temp = TokenClassificationOutputEditor(
                            prediction_rule_based_list[prediction_rule_based_list_index])
                        temp.start_char = prediction_list[prediction_list_index].exclusive_end_char
                        prediction_rule_based_list[
                            prediction_rule_based_list_index
                        ] = temp.render()
                    else:
                        prediction_rule_based_list_index += 1

        while prediction_rule_based_list_index < len(prediction_rule_based_list):
            prediction_list.append(
                prediction_rule_based_list[prediction_rule_based_list_index]
            )
            prediction_rule_based_list_index += 1

        # assert len(set(map(lambda x: tuple(x.items()), prediction_list_save))) == len(
        #     prediction_list_save
        # )
        # assert len(prediction_list_save) == len(
        #     set(map(lambda x: tuple(x.items()), prediction_list_save)).intersection(
        #         set(map(lambda x: tuple(x.items()), prediction_list))
        #     )
        # )

        assert not (
            bit_map
            & np.logical_not(self.generate_bit_map_prediction(prediction_list, report))
        ).sum()

        self.generate_bit_map_prediction(prediction_list, report)
        return prediction_list


    @staticmethod
    def generate_bit_map_prediction(prediction_list: List[TokenClassificationOutput], report):
        bit_map = np.zeros(len(report), dtype=int)

        for prediction in prediction_list:
            bit_map[prediction.start_char: prediction.exclusive_end_char] = 1

        return bit_map


    def fuse_neighbor_predictions_from_the_same_class(self,
                                                      prediction_list: List[TokenClassificationOutput],
                                                      report: str):
        index = 0

        bit_map = self.generate_bit_map_prediction(prediction_list, report)

        while index < len(prediction_list) - 1:
            if prediction_list[index].exclusive_end_char >= prediction_list[index + 1].start_char:
                raise Exception("overlapping")

            assert prediction_list[index].exclusive_end_char < prediction_list[index + 1].start_char

            if prediction_list[index].label != prediction_list[index + 1].label:
                index += 1

            elif all(
                [
                    char in self.characters_that_can_be_skipped
                    for char in report[
                        prediction_list[index].exclusive_end_char : prediction_list[index + 1].start_char
                    ]
                ]
            ):
                assert (
                    prediction_list[index].label
                    == prediction_list[index + 1].label
                )

                temp = TokenClassificationOutputEditor(prediction_list[index])
                temp.score = (prediction_list[index].score
                              * (prediction_list[index].exclusive_end_char - prediction_list[index].start_char)
                              + prediction_list[index + 1].score
                              * (
                                      prediction_list[index + 1].exclusive_end_char
                                      - prediction_list[index + 1].start_char
                              )
                              ) / (
                                     (prediction_list[index].exclusive_end_char - prediction_list[index].start_char)
                                     + (
                                             prediction_list[index + 1].exclusive_end_char
                                             - prediction_list[index + 1].start_char
                                     )
                             )
                temp.exclusive_end_char = prediction_list[index + 1].exclusive_end_char
                prediction_list[index] = temp.render()

                prediction_list.pop(index + 1)

            else:
                index += 1

        assert not (
            bit_map
            & np.logical_not(self.generate_bit_map_prediction(prediction_list, report))
        ).sum()

        return prediction_list

    @classmethod
    def create(cls,
               hospitals: List[str],
               vendors: List[str],
               ):
        return cls(
            rule_based_models=[RuleBasedAgeDetector.create(),
                               RuleBasedSimpleModel.create("HOSPITAL", hospitals),
                               RuleBasedSimpleModel.create("VENDOR", vendors, add_at=True)],
            characters_that_can_be_skipped=set(
                [
                    "\t",
                    "\n",
                    "\r",
                    "\x0b",
                    "\x0c",
                    " ",
                ]
            )
        )
