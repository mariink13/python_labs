def flatten(mat: list[list | tuple]) -> list:
    result = []
    for row in mat:
        if isinstance(
            row, (list, tuple)
        ):  # проверяю является ли row списком или кортежем
            result.extend(row)  # добавляю элементы row по отдельгости в новый список
        else:
            raise TypeError
    return result


print(flatten([[1, 2], [3, 4]]))
print(flatten([[1, 2], (3, 4, 5)]))
print(flatten([[1], [], [2, 3]]))
print(flatten([[1, 2], "ab"]))
