"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors

    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)

    products = np.matmul(matrices, vectors)
    return np.sum(products, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold

    matrix = np.asarray(matrix)
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)

    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)

    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = (
        data.rows,
        data.columns,
        data.mean,
        data.std,
        data.seed,
    )

    rng = np.random.default_rng(seed)

    matrix = rng.normal(
        loc=mean,
        scale=std,
        size=(rows, columns),
    )

    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)

    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)

    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances,
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = (
        data.rows,
        data.columns,
        data.first,
        data.second,
    )

    row_indices, column_indices = np.indices((rows, columns))

    return np.where(
        (row_indices + column_indices) % 2 == 0,
        first,
        second,
    )


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty(
        (image_height, image_width, 3),
        dtype=np.uint8,
    )

    image[:] = background_color

    center_y = image_height // 2
    center_x = image_width // 2

    y_start = center_y - height // 2
    x_start = center_x - width // 2

    y_end = y_start + height
    x_end = x_start + width

    image[y_start:y_end, x_start:x_end] = shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_a, semi_axis_b = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color

    image = np.empty(
        (image_height, image_width, 3),
        dtype=np.uint8,
    )

    image[:] = background_color

    center_x = (image_width - 1) / 2
    center_y = (image_height - 1) / 2

    y, x = np.ogrid[:image_height, :image_width]

    ellipse_mask = (
        ((x - center_x) ** 2) / (semi_axis_a ** 2)
        + ((y - center_y) ** 2) / (semi_axis_b ** 2)
        <= 1
    )

    image[ellipse_mask] = shape_color

    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window

    values = np.asarray(values, dtype=float)

    mean = np.mean(values)
    variance = np.var(values)
    std = np.std(values)

    local_maxima_indices = np.where(
        (values[1:-1] > values[:-2])
        & (values[1:-1] > values[2:])
    )[0] + 1

    local_minima_indices = np.where(
        (values[1:-1] < values[:-2])
        & (values[1:-1] < values[2:])
    )[0] + 1

    moving_average = np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=local_maxima_indices,
        local_minima_indices=local_minima_indices,
        moving_average=moving_average,
    )

def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count

    labels = np.asarray(labels, dtype=int)

    if class_count is None:
        if len(labels) == 0:
            class_count = 0
        else:
            class_count = int(np.max(labels)) + 1

    result = np.zeros(
        (len(labels), class_count),
        dtype=int,
    )

    if len(labels) > 0:
        result[np.arange(len(labels)), labels] = 1

    return result