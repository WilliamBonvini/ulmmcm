from enum import Enum


class Models(Enum):
    MMPN = "mmpn"


class DirValues(Enum):
    VARIANT = "variant"


class ClassValues(Enum):
    CIRCLES = "circles"
    HOMOGRAPHIES = "homographies"


class Options(Enum):
    TRAIN = "train"
    TEST = "test"


class TrainParams(Enum):
    ALL_NOISE_PERCENTAGES = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    ALL_OUTLIERS_RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    OUTLIERS_RATE_25 = 0.25
    OUTLIERS_RATE_50 = 0.50
    OUTLIERS_RATE_60 = 0.60
    NOISE_PERCENTAGE_1 = 0.01

