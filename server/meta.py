from enum import Enum


class InsulinType(str, Enum):
    basal = "basal"
    shot = "shot"
    premix = "premix"


class InsulinPoint(str, Enum):
    morning = "morning"
    nooning = "nooning"
    evening = "evening"
    bedtime = "bedtime"


class Sugar7Point(str, Enum):
    pre_morning = "pre_morning"
    post_morning = "post_morning"

    pre_nooning = "pre_nooning"
    post_nooning = "post_nooning"

    pre_evening = "pre_evening"
    post_evening = "post_evening"

    bedtime = "bedtime"
