
# The state model describes the dimensions of the grid and provides methods to transfrom a pose (x, y, h)
# or a position (x, y) into a state i or a sensor reading r and/or vice versa. x runs over rows, y over
# columns. "State" means the index coding for a pose, "reading" means the index coding for a position.
#
# State 0 = Pose <x=0,y=0, h=0> is the upper left corner of the grid direction "NORTH", rows run then down,
# columns to the right, headings clockwise around the cell.
#
# Seen from above, this corresponds to a robot coordinate frame, where x is "forward", and y "left".
# Seen from the side, you get the "regular" coordinate system with the x-axis running to the right and
# the y-axis running "up".

import numpy as np
import matplotlib.pyplot as plt
import random

class StateModel:

    def __init__(self, rows, cols):
        self.__rows = rows
        self.__cols = cols
        self.__head = 4
        self.__num_states = rows*cols*4
        self.__num_readings = rows*cols+1

    def state_to_pose(self, s: int) -> (int, int, int):
        x = s // (self.__cols * self.__head)
        y = (s - x * self.__cols * self.__head) // self.__head
        h = s % self.__head

        return x, y, h;

    def pose_to_state(self, x: int, y: int, h: int) -> int:
        return x * self.__cols * self.__head + y * self.__head + h

    def state_to_position(self, s: int) -> (int, int):
        x = s // (self.__cols * self.__head)
        y = (s - x * self.__cols * self.__head) // self.__head

        return x, y

    def reading_to_position(self, r: int) -> (int, int):
        x = r // self.__cols
        y = r % self.__cols

        return x, y

    def position_to_reading(self, x: int, y: int) -> int:
        return x * self.__cols + y

    def state_to_reading(self, s: int) -> int:
        return s // self.__head

    # Note that a reading contains less information than a state, i.e., this will always give the state
    # corresponding to heading=0 (NORTH) in the cell that corresponds to "reading r"
    def reading_to_ref_state(self, r: int) -> int:
        return r * self.__head

    def get_grid_dimensions(self) -> (int, int, int):
        return self.__rows, self.__cols, self.__head

    def get_num_of_states(self) -> int:
        return self.__num_states

    def get_num_of_readings(self) -> int:
        return self.__num_readings

