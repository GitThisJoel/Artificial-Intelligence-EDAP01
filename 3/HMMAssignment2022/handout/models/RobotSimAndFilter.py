import random
import numpy as np

from models import TransitionModel, ObservationModel, StateModel

# Add your Robot Simulator here

# REMEBER: X IS NOT COLS ITS ROWS (for some reason?)
class RobotSim:
    def __init__(self, sm: StateModel, tm: TransitionModel):  # true_state: int
        self.__sm = sm
        self.__tm = tm
        # self.__true_state = true_state

        self.__rows, self.__cols, _ = self.__sm.get_grid_dimensions()
        print("Hello from RobotSim")

    def __possible_moves(self, col, row):
        # North = 0
        # East = 1
        # South = 2
        # West = 3
        possible = []
        if row > 0:
            possible.append(0)
        if col < self.__cols - 1:
            possible.append(1)
        if row < self.__rows - 1:
            possible.append(2)
        if col > 0:
            possible.append(3)
        return possible

    def __move_point(self, col, row, h):
        if h == 0:
            return col, row - 1
        if h == 1:
            return col + 1, row
        if h == 2:
            return col, row + 1
        if h == 3:
            return col - 1, row

    def next_state(self, true_state: int):
        # instead of x, y; col, row is used
        row, col = self.__sm.state_to_position(true_state)
        poss_moves = self.__possible_moves(col, row)

        move_probs = {}
        for m in poss_moves:
            col_m, row_m = self.__move_point(col, row, m)
            # check all the headings
            # do not care to filter those that are not possible
            for h in range(0, 4):
                new_state = self.__sm.pose_to_state(row_m, col_m, h)
                prob = self.__tm.get_T_ij(true_state, new_state)
                if prob != 0:
                    move_probs[new_state] = prob

        ms = list(move_probs.keys())
        ps = list(move_probs.values())
        return np.random.choice(ms, p=ps)

    def __gen_LsN(self, col, row):
        Ls = []
        Ls2 = []
        for dcol in range(-2, 3):
            for drow in range(-2, 3):
                if dcol == drow == 0:
                    continue
                if 0 <= col + dcol < self.__cols and 0 <= row + drow < self.__rows:
                    if dcol % 2 == 0 or drow % 2 == 0:
                        Ls2.append((col + dcol, row + drow))
                    else:
                        Ls.append((col + dcol, row + drow))
        return Ls, Ls2

    # returns a reading!
    def sense(self, true_state: int) -> int:
        prob_L = 0.1  # i.e. true locations is reported
        prob_n_Ls = 0.05
        prob_n_Ls2 = 0.025

        row, col = self.__sm.state_to_position(true_state)
        Ls, Ls2 = self.__gen_LsN(col, row)
        # prob_nothing = 1 - prob_L - len(Ls) * prob_n_Ls - len(Ls2) * prob_n_Ls2
        # p = 1 - max(prob_L, len(Ls) * prob_n_Ls, len(Ls2) * prob_n_Ls2)

        prob = random.random()
        if prob <= prob_L:
            row, col, _ = self.__sm.state_to_pose(true_state)
            best_pos = (col, row)
        elif prob <= len(Ls) * prob_n_Ls + prob_L:
            best_pos = random.choice(Ls)
        elif prob <= len(Ls2) * prob_n_Ls2 + len(Ls) * prob_n_Ls + prob_L:
            best_pos = random.choice(Ls2)
        else:
            return None

        # best_pos = (col, row)
        return self.__sm.position_to_reading(best_pos[1], best_pos[0])  # row, col


# Add your Filtering approach here (or within the Localizer, that is your choice!)
#
# f_1:0 = p(X_0),
# f_1:t+1 = alpha * O_e_t+1 * T^T * f_1:t
class HMMFilter:
    def __init__(self, sm: StateModel, tm: TransitionModel, om: ObservationModel):
        self.__tm = tm
        self.__om = om
        self.__sm = sm
        print("Hello from HMMFilter")

    def forward_filter(self, sense_pos, f):
        reading = None
        if not sense_pos == None:
            reading = self.__sm.position_to_reading(
                *self.__sm.reading_to_position(sense_pos)
            )

        O_e_tp1 = self.__om.get_o_reading(reading)
        T_T = self.__tm.get_T_transp()
        fp1 = np.dot(O_e_tp1, np.dot(T_T, f))

        best_pos = None
        max_proba = -1
        for i in range(len(fp1)):
            if fp1[i] > max_proba:
                max_proba = fp1[i]
                best_pos = self.__sm.state_to_position(i)

        fp1 /= sum(fp1)  # np.linalg.norm(fp1)
        return fp1, best_pos
