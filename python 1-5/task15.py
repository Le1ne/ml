from typing import List


def hello(name: str = None) -> str:
    if name is None or name == '':
        return "Hello!"
    else:
        return f"Hello, {name}!"


def int_to_roman(num: int) -> str:
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    res = ""

    for i in range(len(val)):
        while num >= val[i]:
            num -= val[i]
            res += syms[i]

    return res


def longest_common_prefix(strs_input: List[str]) -> str:
    if not strs_input:
        return ""

    processed_strs = [s.lstrip() for s in strs_input]
    min_str = min(processed_strs, key=len)

    for i in range(len(min_str)):
        cur = min_str[i]
        for s in processed_strs:
            if s[i] != cur:
                return min_str[:i]

    return min_str


def primes() -> int:
    is_prime = True
    n = 2
    while True:
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                is_prime = False
                break
        if is_prime:
            yield n
        is_prime = True
        n += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, sum_spent):
        if self.total_sum >= sum_spent:
            self.total_sum -= sum_spent
            print(f"You spent {sum_spent} dollars.")
        else:
            raise ValueError("Not enough money to spend sum_spent dollars.")

    def __str__(self):
        return "To learn the balance call balance."

    @property
    def balance(self):
        if self.balance_limit is not None:
            if self.balance_limit > 0:
                self.balance_limit -= 1
                return self.total_sum
            else:
                raise ValueError("Balance check limits exceeded.")
        else:
            return self.total_sum

    def put(self, sum_put: int):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars")

    def __add__(self, other):
        return BankCard(self.total_sum + other.total_sum, max(self.balance_limit, other.balance_limit))
