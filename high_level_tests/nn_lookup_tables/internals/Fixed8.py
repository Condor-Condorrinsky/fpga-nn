
class Fixed8:
    BITS = 8

    def __init__(self, internal_bits: int = 0) -> None:
        self.__bits = internal_bits
        self.__decimal_pos = self.BITS / 2 # Half - half split for a fresh variable initialization

    @classmethod
    def from_integer(cls, integer: int) -> "Fixed8":
        return cls(integer)

    @classmethod
    def from_2_integers(cls, integer_part: int, fractional_part) -> "Fixed8":
        pass

    @classmethod
    def from_float(cls, floating: float) -> "Fixed8":
        pass

    @staticmethod
    def __is_power_of_2(n: int) -> bool:
        return (n & (n - 1) == 0) and n != 0

if __name__ == '__main__':
    fixed_8 = Fixed8()
