"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = "aeiou"
    return sum(1 for char in text.lower() if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    count = 0

    while number >= 10:
        product = 1

        for digit in str(number):
            product *= int(digit)

        number = product
        count += 1

    return count


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value

    if number == 1:
        return "(1)"

    result = []
    factor = 2

    while factor * factor <= number:
        power = 0

        while number % factor == 0:
            number //= factor
            power += 1

        if power == 1:
            result.append(f"({factor})")
        elif power > 1:
            result.append(f"({factor}**{power})")

        factor += 1

    if number > 1:
        result.append(f"({number})")

    return "".join(result)

def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value

    total = 0
    k = 0

    while total < cube_count:
        k += 1
        total += k ** 2

    if total == cube_count:
        return k

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = str(number)
    length = len(digits)
    middle = length // 2

    if length % 2 == 1:
        left = digits[:middle]
        right = digits[middle + 1:]
    else:
        left = digits[:middle - 1]
        right = digits[middle + 1:]

    left_sum = sum(int(digit) for digit in left)
    right_sum = sum(int(digit) for digit in right)

    return left_sum == right_sum