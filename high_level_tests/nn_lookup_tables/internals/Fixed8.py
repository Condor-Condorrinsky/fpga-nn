
class Fixed8:
    BITS = 8
    # Signed 2's complement
    LOWER_BOUND = - 2 ** (BITS - 1)
    UPPER_BOUND = 2 ** (BITS - 1) - 1
    TOTAL_VALS = 2 ** BITS

    def __init__(self, internal_bits: int = 0, decimal_pos: int | None = None) -> None:

        if internal_bits < 0 or internal_bits > self.TOTAL_VALS:
            raise ValueError("Internal bits out of range for Fixed8")
        self.__bits = internal_bits

        if decimal_pos is None:
            self.__decimal_pos = self.BITS / 2 # Half - half split for a fresh variable initialization
        elif decimal_pos < 0 or decimal_pos > self.BITS - 1 :
            raise ValueError("Decimal position out of range for Fixed8")
        else:
            self.__decimal_pos = decimal_pos


    @classmethod
    def from_integer(cls, integer: int) -> "Fixed8":
        if integer < Fixed8.LOWER_BOUND:
            return cls(internal_bits=Fixed8.LOWER_BOUND, decimal_pos=0)
        if integer > Fixed8.UPPER_BOUND:
            return cls(internal_bits=Fixed8.UPPER_BOUND, decimal_pos=0)
        return cls(internal_bits=integer, decimal_pos=0)

    @classmethod
    def from_2_integers(cls, integer_part: int, fractional_part) -> "Fixed8":
        pass

    @classmethod
    def from_float(cls, floating: float) -> "Fixed8":
        pass

    @staticmethod
    def __is_power_of_2(n: int) -> bool:
        return (n & (n - 1) == 0) and n != 0

    def to_decimal(self) -> int:
        pass

if __name__ == '__main__':
    fixed_8 = Fixed8()
