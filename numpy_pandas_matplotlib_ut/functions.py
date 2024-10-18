from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    len_d = min(len(X), len(X[0]))
    res = 0
    is_non_neg = False

    for i in range(len_d):
        if X[i][i] >= 0:
            is_non_neg = True
            res += X[i][i]

    if not is_non_neg:
        return -1

    return res


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    x.sort(), y.sort()

    if x != y:
        return False

    return True


def max_prod_mod_3(x: List[int]) -> int:
    res = -1

    for i in range(len(x) - 1):
        if (x[i] % 3 == 0 or x[i + 1] % 3 == 0) and x[i] * x[i + 1] > res:
            res = x[i] * x[i + 1]

    return res


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    height, width, num_channels = len(image), len(image[0]), len(image[0][0])
    res_matrix = [[0] * width for _ in range(height)]

    for i in range(height):
        for j in range(width):
            value = 0
            for channel in range(num_channels):
                value += image[i][j][channel] * weights[channel]
            res_matrix[i][j] = value

    return res_matrix


def rle_scalar(x: List[List[int]], y: List[List[int]]) -> int:
    len_x = sum([elem[1] for elem in x])
    len_y = sum([elem[1] for elem in y])

    if len_x != len_y:
        return -1

    i_x, j_x, i_y, j_y = 0, 0, 0, 0
    scalar = 0

    while i_x < len(x) and i_y < len(y):
        while j_x < x[i_x][1] and j_y < y[i_y][1]:
            scalar += x[i_x][0] * y[i_y][0]
            j_x += 1
            j_y += 1
        if j_x == x[i_x][1]:
            j_x = 0
            i_x += 1
        if j_y == y[i_y][1]:
            j_y = 0
            i_y += 1

    return scalar


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    m = [[0] * len(Y) for _ in range(len(X))]

    for i in range(len(X)):
        for j in range(len(Y)):
            scalar = sum(x * y for x, y in zip(X[i], Y[j]))
            norm_x = sum(x ** 2 for x in X[i]) ** 0.5
            norm_y = sum(y ** 2 for y in Y[j]) ** 0.5

            if norm_x == 0 or norm_y == 0:
                m[i][j] = 1
            else:
                m[i][j] = scalar / (norm_x * norm_y)

    return m
