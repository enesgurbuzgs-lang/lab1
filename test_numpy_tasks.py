import numpy as np

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)

from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput,
    BinarizeInput,
    MatrixInput,
    RandomMatrixInput,
    ChessInput,
    RectangleInput,
    EllipseInput,
    TimeSeriesInput,
    OneHotInput,
)


matrices = np.array([
    [[1, 0], [0, 1]],
    [[2, 0], [0, 2]],
])

vectors = np.array([
    [[1], [2]],
    [[3], [4]],
])

result = sum_prod(
    MatrixVectorBatchInput(
        matrices=matrices,
        vectors=vectors
    )
)

expected = np.array([
    [7],
    [10]
])

assert np.array_equal(result, expected)
assert result.shape == (2, 1)

print("1. sum_prod OK")


matrix = np.array([
    [1, 5, 3],
    [7, 2, 5]
])

original = matrix.copy()

result = binarize(
    BinarizeInput(
        matrix=matrix,
        threshold=3
    )
)

expected = np.array([
    [0, 1, 0],
    [1, 0, 1]
])

assert np.array_equal(result, expected)
assert np.array_equal(matrix, original)

print("2. binarize OK")


matrix = np.array([
    [1, 1, 2],
    [2, 3, 3]
])

data = MatrixInput(matrix=matrix)

assert unique_rows(data) == [
    [1, 2],
    [2, 3]
]

assert unique_columns(data) == [
    [1, 2],
    [1, 3],
    [2, 3]
]

print("3. unique_rows / unique_columns OK")


stats = matrix_statistics(
    RandomMatrixInput(
        rows=3,
        columns=4,
        mean=0.0,
        std=1.0,
        seed=42
    )
)

assert stats.matrix.shape == (3, 4)

assert np.allclose(
    stats.row_means,
    np.mean(stats.matrix, axis=1)
)

assert np.allclose(
    stats.column_means,
    np.mean(stats.matrix, axis=0)
)

assert np.allclose(
    stats.row_variances,
    np.var(stats.matrix, axis=1)
)

assert np.allclose(
    stats.column_variances,
    np.var(stats.matrix, axis=0)
)

print("4. matrix_statistics OK")


result = chess(
    ChessInput(
        rows=3,
        columns=4,
        first=0,
        second=1
    )
)

expected = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
])

assert np.array_equal(result, expected)

print("5. chess OK")


rectangle = draw_rectangle(
    RectangleInput(
        width=4,
        height=2,
        image_height=6,
        image_width=8,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )
)

assert rectangle.shape == (6, 8, 3)

red_pixels = np.all(
    rectangle == np.array([255, 0, 0]),
    axis=2
)

assert np.sum(red_pixels) == 8

print("6. draw_rectangle OK")


ellipse = draw_ellipse(
    EllipseInput(
        semi_axis_x=2,
        semi_axis_y=1,
        image_height=7,
        image_width=7,
        shape_color=(255, 255, 255),
        background_color=(0, 0, 0)
    )
)

assert ellipse.shape == (7, 7, 3)

assert np.any(
    np.all(
        ellipse == np.array([255, 255, 255]),
        axis=2
    )
)

print("6b. draw_ellipse OK")


series = np.array([
    1, 3, 1, 4, 2
])

stats = analyze_time_series(
    TimeSeriesInput(
        values=series,
        window=3
    )
)

assert np.isclose(
    stats.mean,
    np.mean(series)
)

assert np.isclose(
    stats.variance,
    np.var(series)
)

assert np.isclose(
    stats.std,
    np.std(series)
)

assert np.array_equal(
    stats.local_maxima_indices,
    np.array([1, 3])
)

assert np.array_equal(
    stats.local_minima_indices,
    np.array([2])
)

expected_moving_average = np.array([
    (1 + 3 + 1) / 3,
    (3 + 1 + 4) / 3,
    (1 + 4 + 2) / 3
])

assert np.allclose(
    stats.moving_average,
    expected_moving_average
)

print("7. analyze_time_series OK")


result = one_hot(
    OneHotInput(
        labels=[0, 2, 3, 0],
        class_count=None
    )
)

expected = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
])

assert np.array_equal(result, expected)

result2 = one_hot(
    OneHotInput(
        labels=[0, 2],
        class_count=5
    )
)

assert result2.shape == (2, 5)

print("8. one_hot OK")


print("\nALL NUMPY TESTS WERE PASSED SUCCESSFULLY!")


print("\nAdditional boundary tests are beginning...")


result = sum_prod(
    MatrixVectorBatchInput(
        matrices=np.array([[[5]]]),
        vectors=np.array([[[2]]])
    )
)

assert np.array_equal(result, np.array([[10]]))
assert result.shape == (1, 1)

print("1. sum_prod boundary OK")


result = binarize(
    BinarizeInput(
        matrix=np.array([
            [3, 4],
            [2, 3]
        ]),
        threshold=3
    )
)

expected = np.array([
    [0, 1],
    [0, 0]
])

assert np.array_equal(result, expected)

print("2. binarize boundary OK")


data = MatrixInput(
    matrix=np.array([[7]])
)

assert unique_rows(data) == [[7]]
assert unique_columns(data) == [[7]]

print("3. unique rows/columns boundary OK")


stats = matrix_statistics(
    RandomMatrixInput(
        rows=1,
        columns=1,
        mean=5.0,
        std=0.0,
        seed=42
    )
)

assert stats.matrix.shape == (1, 1)
assert np.allclose(stats.matrix, [[5.0]])

assert np.allclose(
    stats.row_means,
    [5.0]
)

assert np.allclose(
    stats.column_means,
    [5.0]
)

assert np.allclose(
    stats.row_variances,
    [0.0]
)

assert np.allclose(
    stats.column_variances,
    [0.0]
)

print("4. matrix_statistics boundary OK")


result = chess(
    ChessInput(
        rows=1,
        columns=1,
        first=9,
        second=4
    )
)

assert np.array_equal(
    result,
    np.array([[9]])
)

print("5. chess boundary OK")


result = draw_rectangle(
    RectangleInput(
        width=1,
        height=1,
        image_height=1,
        image_width=1,
        shape_color=(255, 0, 0),
        background_color=(0, 0, 0)
    )
)

assert result.shape == (1, 1, 3)
assert np.array_equal(
    result[0, 0],
    np.array([255, 0, 0])
)

print("6. draw_rectangle boundary OK")


result = draw_ellipse(
    EllipseInput(
        semi_axis_x=1,
        semi_axis_y=1,
        image_height=3,
        image_width=3,
        shape_color=(255, 255, 255),
        background_color=(0, 0, 0)
    )
)

assert result.shape == (3, 3, 3)

assert np.array_equal(
    result[1, 1],
    np.array([255, 255, 255])
)

print("6b. draw_ellipse boundary OK")


series = np.array([2, 4, 6])

stats = analyze_time_series(
    TimeSeriesInput(
        values=series,
        window=1
    )
)

assert np.allclose(
    stats.moving_average,
    series
)

assert len(stats.local_maxima_indices) == 0
assert len(stats.local_minima_indices) == 0

print("7. analyze_time_series boundary OK")


result = one_hot(
    OneHotInput(
        labels=[0, 1],
        class_count=4
    )
)

expected = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

assert np.array_equal(result, expected)
assert result.shape == (2, 4)

print("8. one_hot boundary OK")


print("\nALL ADDITIONAL LIMIT TESTS WERE PASSED SUCCESSFULLY!")