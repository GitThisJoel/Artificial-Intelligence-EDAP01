# The transition model contains the transition matrix and some methods for convenience,
# including transposition

import numpy as np
import matplotlib.pyplot as plt
import random

import models.StateModel


class TransitionModel:
    def __init__(self, stateModel):
        self.__sm = stateModel
        self.__rows, self.__cols, self.__head = self.__sm.get_grid_dimensions()

        self.__dim = self.__rows * self.__cols * self.__head

        self.__matrix = np.zeros(shape=(self.__dim, self.__dim), dtype=float)
        for i in range(self.__dim):
            x, y, h = self.__sm.state_to_pose(i)
            for j in range(self.__dim):
                nx, ny, nh = self.__sm.state_to_pose(j)

                if abs(x - nx) + abs(y - ny) == 1 and (
                    nh == 0
                    and nx == x - 1
                    or nh == 1
                    and ny == y + 1
                    or nh == 2
                    and nx == x + 1
                    or nh == 3
                    and ny == y - 1
                ):

                    if nh == h:
                        self.__matrix[i, j] = 0.7

                    else:
                        if (
                            x != 0
                            and x != self.__rows - 1
                            and y != 0
                            and y != self.__cols - 1
                        ):
                            self.__matrix[i, j] = 0.1

                        elif (
                            h == 0
                            and x == 0
                            and y != 0
                            and y != self.__cols - 1
                            or h == 1
                            and x != 0
                            and x != self.__rows - 1
                            and y == self.__cols - 1
                            or h == 2
                            and x == self.__rows - 1
                            and y != 0
                            and y != self.__cols - 1
                            or h == 3
                            and x != 0
                            and x != self.__rows - 1
                            and y == 0
                        ):

                            self.__matrix[i, j] = 1.0 / 3.0

                        elif (
                            h != 0
                            and x == 0
                            and y != 0
                            and y != self.__cols - 1
                            or h != 1
                            and x != 0
                            and x != self.__rows - 1
                            and y == self.__cols - 1
                            or h != 2
                            and x == self.__rows - 1
                            and y != 0
                            and y != self.__cols - 1
                            or h != 3
                            and x != 0
                            and x != self.__rows - 1
                            and y == 0
                        ):

                            self.__matrix[i, j] = 0.15

                        elif (
                            (h == 0 or h == 3)
                            and (nh == 1 or nh == 2)
                            and x == 0
                            and y == 0
                            or (h == 0 or h == 1)
                            and (nh == 2 or nh == 3)
                            and x == 0
                            and y == self.__cols - 1
                            or (h == 1 or h == 2)
                            and (nh == 0 or nh == 3)
                            and x == self.__rows - 1
                            and y == self.__cols - 1
                            or (h == 2 or h == 3)
                            and (nh == 0 or nh == 1)
                            and x == self.__rows - 1
                            and y == 0
                        ):

                            self.__matrix[i, j] = 0.5

                        elif (
                            (h == 1 and nh == 2 or h == 2 and nh == 1)
                            and x == 0
                            and y == 0
                            or (h == 2 and nh == 3 or h == 3 and nh == 2)
                            and x == 0
                            and y == self.__cols - 1
                            or (h == 0 and nh == 1 or h == 1 and nh == 0)
                            and x == self.__rows - 1
                            and y == 0
                            or (h == 0 and nh == 3 or h == 3 and nh == 0)
                            and x == self.__rows - 1
                            and y == self.__cols - 1
                        ):

                            self.__matrix[i, j] = 0.3

        if (
            self.__rows == 1 or self.__cols == 1
        ) and self.__rows * self.__cols != 1:  # if we only have one row or colum in the grid, but more than 1 cells
            for i in range(self.__dim):
                sum = np.sum(self.__matrix[i, :])
                self.__matrix[i, :] = self.__matrix[i, :] / sum

    # retrieve the number of states represented in the matrix
    def get_num_of_states(self) -> int:
        return self.__dim

    # get the probability to go from state i to j
    def get_T_ij(self, i: int, j: int) -> float:
        return self.__matrix[i, j]

    # get the entire matrix (dimensions: nr_of_states x nr_of_states, type float)
    def get_T(self) -> np.array(2):
        print(type(self.__matrix))
        return self.__matrix.copy()

    # get the transposed transition matrix (dimensions: nr_of_states x nr_of_states, type float)
    def get_T_transp(self) -> np.array(2):
        transp = np.transpose(self.__matrix)
        return transp

    # plot matrix as a heat map
    def plot_T(self):
        plt.matshow(self.__matrix)
        plt.colorbar()
        plt.show()
