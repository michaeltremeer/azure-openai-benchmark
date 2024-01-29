import enum


# Create some enums so it's much easier to auto-complete model names/types
class MODEL(enum.Enum):
    GPT_35_TURBO_0613_4K = "GPT_35_TURBO_0613_4K"
    GPT_35_TURBO_0613_16K = "GPT_35_TURBO_0613_16K"
    GPT_35_TURBO_1106_16K = "GPT_35_TURBO_1106_16K"
    GPT_4_0613_8K = "GPT_4_0613_8K"
    GPT_4_0613_32K = "GPT_4_0613_32K"
    GPT_4_TURBO_1106_128K = "GPT_4_TURBO_1106_128K"
    GPT_4_TURBO_VISION_1106 = "GPT_4_TURBO_VISION_1106"


class DEPLOYMENT_TYPE(enum.Enum):
    PAYGO = "PAYGO"
    PTU = "PTU"
