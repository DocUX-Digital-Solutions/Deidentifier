import datetime
from dataclasses import dataclass
import datetime as DT
import re
from typing import List, Dict, Tuple
import pandas as pd
from ml_util import docux_logger
import random
from datetime import datetime
from collections import Counter

from date_utils.full_digits_no_trailing_zeros import (
    convert_date_to_string_full_digits_no_trailing_zeros,
    convert_string_to_date_full_digits_no_trailing_zeros,
)
from date_utils.full_digits_with_trailing_zeros import (
    convert_date_to_string_full_digits_with_trailing_zeros,
    convert_string_to_date_full_digits_with_trailing_zeros,
)
from date_utils.full_digits_no_day import (
    convert_date_to_string_full_digits_no_day,
    convert_string_to_date_full_digits_no_day,
)
from date_utils.full_digits_no_year import (
    convert_date_to_string_full_digits_no_year,
    convert_string_to_date_full_digits_no_year,
)

from date_utils.digits_letters_no_year import (
    convert_date_to_string_digits_letters_no_year,
    convert_string_to_date_digits_letters_no_year,
)
from date_utils.digits_letters_no_day import (
    convert_date_to_string_digits_letters_no_day,
    convert_string_to_date_digits_letters_no_day,
)
from date_utils.digits_letters_with_comma import (
    convert_date_to_string_digits_letters_with_comma,
    convert_string_to_date_digits_letters_with_comma,
)
from date_utils.digits_letters_no_comma import (
    convert_date_to_string_digits_letters_no_comma,
    convert_string_to_date_digits_letters_no_comma,
)
from ml_util.label_tokens import CompiledDocLabels, tally_doc_set_label_values

logger = docux_logger.give_logger()

Date = DT.datetime

_ref_year = datetime.now().year - 11

@dataclass(frozen=True)
class DateConverter:
    UNK_FORMAT = 'OTHER'
    slash_prob = 0.6
    _convert_date_to_string = {
        "full_digits_no_trailing_zeros": convert_date_to_string_full_digits_no_trailing_zeros,
        "full_digits_with_trailing_zeros": convert_date_to_string_full_digits_with_trailing_zeros,
        "full_digits_no_day": convert_date_to_string_full_digits_no_day,
        "full_digits_no_year": convert_date_to_string_full_digits_no_year,
        "digits_letters_no_year": convert_date_to_string_digits_letters_no_year,
        "digits_letters_no_day": convert_date_to_string_digits_letters_no_day,
        "digits_letters_with_comma": convert_date_to_string_digits_letters_with_comma,
        "digits_letters_no_comma": convert_date_to_string_digits_letters_no_comma,
    }
    _convert_string_to_date = {
        "full_digits_no_trailing_zeros": convert_string_to_date_full_digits_no_trailing_zeros,
        "full_digits_with_trailing_zeros": convert_string_to_date_full_digits_with_trailing_zeros,
        "full_digits_no_day": convert_string_to_date_full_digits_no_day,
        "full_digits_no_year": convert_string_to_date_full_digits_no_year,
        "digits_letters_no_year": convert_string_to_date_digits_letters_no_year,
        "digits_letters_no_day": convert_string_to_date_digits_letters_no_day,
        "digits_letters_with_comma": convert_string_to_date_digits_letters_with_comma,
        "digits_letters_no_comma": convert_string_to_date_digits_letters_no_comma,
    }

    @staticmethod
    def date_to_string(in_date: Date,
                       format: str) -> str:
        generated_date = DateConverter._convert_date_to_string[format](in_date)

        if random.random() > DateConverter.slash_prob:
            generated_date = generated_date.replace("/", "-")

        return generated_date

    @staticmethod
    def string_to_date_and_format(date: str) -> Tuple[Date, str]:
        try:
            format = DateConverter._base_parse_date_format(date)
        except:
            logger.warning(f"Failed to identify the format of date: {date}")

            date_format = DateConverter.UNK_FORMAT
            try:
                out_date = pd.to_datetime(date).to_pydatetime().replace(tzinfo=None)
            except:
                logger.warning(f"Failed to parse date with pandas: {date}")
                # Adjust so it can come close to the current time.
                out_date = DT.datetime.strptime(
                    f"1/1/{_ref_year}", "%m/%d/%Y"
                ) + DT.timedelta(days=random.randint(-3650, 3650))
            return out_date, date_format

        else:
            return DateConverter._convert_string_to_date[format](date), format

    @staticmethod
    def string_to_date(date: str) -> Date:
        return DateConverter.string_to_date_and_format(date)[0]

    @staticmethod
    def parse_date_format(date: str) -> str:
        return DateConverter.string_to_date_and_format(date)[1]

    @staticmethod
    def _base_parse_date_format(date: str):
        """Parse the format of a date which is a string
Returns its format as a string
        """
        if bool(re.match(r".*([a-z]|[A-Z]).*", date, flags=re.DOTALL)):
            # It has some letters in it
            if sum(c.isdigit() for c in date) < 4:
                return "digits_letters_no_year"  # 'March 1'
            elif sum(c.isdigit() for c in date) == 4:
                return "digits_letters_no_day"  # 'March 2018'
            elif sum(c.isdigit() for c in date) > 4:
                if "," in date:  # and date[date.index(',') - 1].isdigit():
                    return "digits_letters_with_comma"  # 'March 1, 2018'
                else:
                    return "digits_letters_no_comma"  # '1 March 2018'
        else:
            # no letters
            if (
                    bool(re.match(r"(.*0.{1}\/.*)", date, flags=re.DOTALL))
                    and date.count("/") == 2
            ):
                return "full_digits_with_trailing_zeros"  # '01/02/2020'
            elif (
                    bool(re.match(r"(.+\/.{4})", date, flags=re.DOTALL))
                    or bool(re.match(r"(.{4}\/.+)", date, flags=re.DOTALL))
            ) and date.count("/") == 1:
                return "full_digits_no_day"  # '2/2020'
            elif date.count("/") == 1:
                return "full_digits_no_year"  # '1/2'
            else:
                return "full_digits_no_trailing_zeros"  # '1/1/2020'

    @staticmethod
    def tally_dates_and_formats(report_tallies: List[CompiledDocLabels]) -> Tuple[Dict[Date, int], Dict[str, int]]:
        group_tally = tally_doc_set_label_values(report_tallies, ["DATE"])

        format_tally = {k: 0 for k in DateConverter._convert_date_to_string.keys()}
        date_tally = Counter()
        for raw_date, cnt in group_tally["DATE"].items():
            date, format = DateConverter.string_to_date_and_format(raw_date)
            if format != DateConverter.UNK_FORMAT:
                date_tally[date] += 1
                format_tally[format] += cnt

        return dict(date_tally), format_tally
