# A file for generating point data

import random
import numpy as np


def generate_point_data(number_of_points=100, y_intercept=0, slope=1):
    """A function that will generate points around a given line and label the ones above 1 and bellow 0."""

    point_dataset = []
    for _ in range(number_of_points):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        y_on_line = slope * x + y_intercept
        if y > y_on_line:
            label = 1
        else:
            label = 0
        point_dataset.append(np.array([x, y, label]))

    return point_dataset