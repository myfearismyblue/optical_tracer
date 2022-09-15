__all__ = ['DEBUG',
           'OPT_SYS_DIMENSIONS',
           'OPTICAL_RANGE',
           'QUARTER_PART_IN_MM',
           'TOLL',
           'PERCENT',
           'METRE',
           'CENTIMETRE',
           'MILLIMETRE',
           'NANOMETRE',
           'N0_REFLECTION',
           'NO_REFRACTION', ]

from typing import Tuple

DEBUG = False
OPT_SYS_DIMENSIONS = (-100, 100)
OPTICAL_RANGE: Tuple[int, int] = (380, 780)  # in nanometers
QUARTER_PART_IN_MM = 10 ** (-6) / 4  # used in expressions like 555 nm * 10 ** (-6) / 4 to represent tolerance
TOLL = 10 ** -3  # to use in scipy functions
PERCENT = 0.01
METRE = 1
CENTIMETRE = METRE * 10 ** -2
MILLIMETRE = METRE * 10 ** -3
NANOMETRE = METRE * 10 ** -9
N0_REFLECTION = NO_REFRACTION = False and DEBUG
