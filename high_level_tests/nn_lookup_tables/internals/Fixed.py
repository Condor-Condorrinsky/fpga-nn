
class Fixed:

    def __init__(self, bits: int) -> None:
        pass

    @staticmethod
    def __is_power_of_2(n: int) -> bool:
        return (n & (n - 1) == 0) and n != 0
