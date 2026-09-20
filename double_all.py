def double_all(numbers):
    doubled = []
    for number in numbers:
        doubled.append(number * 2)

    return doubled

numbers = [1, 3, 5]

print(double_all(numbers))