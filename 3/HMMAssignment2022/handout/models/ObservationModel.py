# The observation model contains the diagonals (stored as vectors) of the observation
# matrices for each possible sensor reading
# The last of these vectors contains the probabilities for the sensor to produce nothing

import numpy as np
import matplotlib.pyplot as plt
import random

import models.StateModel

class ObservationModel:
    def __init__(self, stateModel):

        self.__stateModel = stateModel
        self.__rows, self.__cols, self.__head = stateModel.get_grid_dimensions()

        self.__dim = self.__rows * self.__cols * self.__head
        self.__num_readings = self.__rows * self.__cols + 1

        self.__vectors = np.ones(shape=(self.__num_readings, self.__dim))

        for o in range(self.__num_readings - 1):
            sx, sy = self.__stateModel.reading_to_position(o)

            for i in range(self.__dim):
                x, y = self.__stateModel.state_to_position(i)
                self.__vectors[o, i] = 0.0

                if x == sx and y == sy:
                    self.__vectors[o, i] = 0.1
                elif (x == sx + 1 or x == sx - 1) and y == sy:
                    self.__vectors[o, i] = 0.05
                elif (x == sx + 1 or x == sx - 1) and (y == sy + 1 or y == sy - 1):
                    self.__vectors[o, i] = 0.05
                elif x == sx and (y == sy + 1 or y == sy - 1):
                    self.__vectors[o, i] = 0.05
                elif (x == sx + 2 or x == sx - 2) and (y == sy or y == sy + 1 or y == sy - 1):
                    self.__vectors[o, i] = 0.025
                elif (x == sx + 2 or x == sx - 2) and (y == sy + 2 or y == sy - 2):
                    self.__vectors[o, i] = 0.025
                elif (x == sx or x == sx + 1 or x == sx - 1) and (y == sy + 2 or y == sy - 2):
                    self.__vectors[o, i] = 0.025

                self.__vectors[self.__num_readings - 1, i] -= self.__vectors[o, i];  # sensor reading "nothing"

    # get the number of possible sensor readings (rows * columns + 1)
    def get_nr_of_readings(self) -> int:
        return self.__num_readings

    # get the probability for the sensor to have produced reading "reading" when in state "state"
    def get_o_reading_state(self, reading: int, i: int) -> float:
        if reading == None : reading = self.__num_readings-1
        return self.__vectors[reading, i]

    # get the diagonale matrix O_reading with probabilities of the states i, i=0...nrOfStates-1 
    # to have produced reading "reading", returns a 2d-float array
    # use None for "no reading"
    def get_o_reading(self, reading: int) -> np.array(2):
        if (reading == None): reading = self.__num_readings - 1
        return np.diag( self.__vectors[reading, :])

    # plot the vectors as heat map(s)
    def plot_o_diags(self):
        plt.matshow(self.__vectors)
        plt.colorbar()
        plt.show()
