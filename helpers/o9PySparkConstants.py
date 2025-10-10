from o9Reference.spark_utils.common_utils import get_clean_string, is_dimension

from helpers.o9Constants import o9Constants


class o9PySparkConstantsMeta(type):
    def __getattr__(cls, name):
        value = getattr(o9Constants, name)
        if is_dimension(input_string=value):
            return get_clean_string(input_string=value)
        else:
            return value


class o9PySparkConstants(metaclass=o9PySparkConstantsMeta):
    pass


if __name__ == "__main__":
    # Example usage:
    print(o9Constants.STAT_ITEM)  # Output: Item.[Stat Item]

    # Access constants from o9PySparkConstants
    print(o9PySparkConstants.STAT_ITEM)  # Output: Item_StatItem

    # Example usage:
    print(o9Constants.STAT_ACTUAL)  # Output: Stat Actual

    # Access constants from o9PySparkConstants
    print(o9PySparkConstants.STAT_ACTUAL)  # Output: Stat Actual
