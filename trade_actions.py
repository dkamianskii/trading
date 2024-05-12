from enum import Enum


class TradeAction(Enum):
    NONE = 0
    LONG = 1
    CLOSE_LONG = 2
    SHORT = 3
    CLOSE_SHORT = 4
