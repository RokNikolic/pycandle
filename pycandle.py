# A Python library meant for learning machine learning

import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_point_data


def relu(value):
    return np.maximum(0, value)


def step(value):
    if value > 0:
        return 1
    else:
        return 0


class Perceptron:
    bias = 1
    learning_rate = 0.1
    def __init__(self, num_of_inputs):
        self.weights = np.random.randn(num_of_inputs + 1)

    def feedforward(self, data):
        data_with_bias = np.append(data, self.bias)
        weighted_inputs = np.dot(data_with_bias, self.weights)
        summed_input = np.sum(weighted_inputs)
        return step(summed_input)

    def train(self, data, correct_answer):
        guess = self.feedforward(data)
        error = correct_answer - guess
        print(error)
        self.weights += np.append(data, self.bias) * error * self.learning_rate


if __name__ == "__main__":
    line_y_intercept = 0.5
    line_slope = 2
    point_dataset = generate_point_data(y_intercept=line_y_intercept, slope=line_slope)
    x_dataset = [point[0] for point in point_dataset]
    y_dataset = [point[1] for point in point_dataset]

    p1 = Perceptron(2)

    for point in point_dataset:
        inputs = point[0:2]
        target = point[2]
        p1.train(inputs, target)

    learned_y_intercept = float((-p1.weights[2]/p1.weights[1]) * p1.bias)
    learned_slope = float(-p1.weights[0]/p1.weights[1])

    plt.axline((0, learned_y_intercept), slope=learned_slope, color='red')
    plt.scatter(x_dataset, y_dataset)
    plt.show()