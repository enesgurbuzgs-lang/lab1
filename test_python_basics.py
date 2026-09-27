from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)

from grader_contracts.python_basics import (
    TextInput,
    PositiveIntegerInput,
    VectorPairInput,
)


assert count_vowels(TextInput(value="hello")) == 2
assert count_vowels(TextInput(value="AEIOU")) == 5
assert count_vowels(TextInput(value="xyz")) == 0
assert count_vowels(TextInput(value="")) == 0

print("1. count_vowels OK")


assert has_unique_characters(TextInput(value="abc")) is True
assert has_unique_characters(TextInput(value="hello")) is False
assert has_unique_characters(TextInput(value="")) is True
assert has_unique_characters(TextInput(value="Aa")) is True

print("2. has_unique_characters OK")


assert count_one_bits(PositiveIntegerInput(value=1)) == 1
assert count_one_bits(PositiveIntegerInput(value=2)) == 1
assert count_one_bits(PositiveIntegerInput(value=7)) == 3
assert count_one_bits(PositiveIntegerInput(value=13)) == 3

print("3. count_one_bits OK")


assert multiplicative_persistence(PositiveIntegerInput(value=39)) == 3
assert multiplicative_persistence(PositiveIntegerInput(value=4)) == 0
assert multiplicative_persistence(PositiveIntegerInput(value=999)) == 4
assert multiplicative_persistence(PositiveIntegerInput(value=10)) == 1

print("4. multiplicative_persistence OK")


assert mse(VectorPairInput(
    predicted=[1, 2, 3],
    expected=[1, 2, 3]
)) == 0

assert mse(VectorPairInput(
    predicted=[1, 2],
    expected=[3, 4]
)) == 4

print("5. mse OK")


assert prime_factorization(PositiveIntegerInput(value=12)) == "(2**2)(3)"
assert prime_factorization(PositiveIntegerInput(value=18)) == "(2)(3**2)"
assert prime_factorization(PositiveIntegerInput(value=7)) == "(7)"
assert prime_factorization(PositiveIntegerInput(value=86240)) == "(2**5)(5)(7**2)(11)"

print("6. prime_factorization OK")


assert pyramid(PositiveIntegerInput(value=1)) == 1
assert pyramid(PositiveIntegerInput(value=5)) == 2
assert pyramid(PositiveIntegerInput(value=14)) == 3
assert pyramid(PositiveIntegerInput(value=30)) == 4
assert pyramid(PositiveIntegerInput(value=10)) == "It is impossible"

print("7. pyramid OK")


assert is_balanced_number(PositiveIntegerInput(value=1234006)) is True
assert is_balanced_number(PositiveIntegerInput(value=123456)) is False
assert is_balanced_number(PositiveIntegerInput(value=12321)) is True
assert is_balanced_number(PositiveIntegerInput(value=12345)) is False

print("8. is_balanced_number OK")


print("\nALL TESTS WERE PASSED SUCCESSFULLY!")