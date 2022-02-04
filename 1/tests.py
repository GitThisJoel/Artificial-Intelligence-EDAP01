from logging import NullHandler
import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv
import copy  # use this

import json
import random

from skeleton import evaluate_board, alpha_beta, win_next_turn

env: ConnectFourEnv = gym.make("ConnectFour-v0")


INF = 10 ** 20


def is_winning(board, x, y):
    if board[y][x] == 0:
        return False

    # row
    for i in range(0, 4):  # 0 to 3
        if x - i >= 0 and x - i + 3 < len(board[y]):
            if 4 * board[y][x] == sum(board[y][x - i : x - i + 4]):
                return True

    # down
    if len(board) - y >= 4:
        if 4 * board[y][x] == sum([row[x] for row in board[y : y + 4]]):
            return True

    # diag x2
    for i in range(0, 4):
        value = 0
        for j in range(0, 4):
            if (x - i >= 0 and x - i + 3 < len(board[y])) and (
                y - i >= 0 and y - i + 3 < len(board)
            ):
                value += board[y - i + j][x - i + j]
        if 4 * board[y][x] == value:
            return True
    return False


def test_is_winning():
    b1 = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ]

    b2 = [
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
        [0, 1, 0, 0],
    ]

    b3 = [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ]

    print("True:", is_winning(b1, 3, 2))
    print("False:", is_winning(b2, 3, 2))
    print("False:", is_winning(b3, 3, 2))

    print()

    print("False:", is_winning(b1, 1, 0))
    print("True:", is_winning(b2, 1, 0))
    print("False:", is_winning(b3, 1, 0))

    print()

    print("False:", is_winning(b1, 0, 0))
    print("False:", is_winning(b2, 0, 0))
    print("True:", is_winning(b3, 0, 0))

    print()

    print("True:", is_winning(b3, 3, 3))
    return


def test_eval():
    b1 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ]

    b2 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]

    b3 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0],
    ]

    b4 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
        [-1, 0, -1, -1, 0, 1, -1],
        [-1, 1, -1, 1, -1, 1, 1],
        [1, 1, 1, -1, 1, 1, -1],
    ]

    b5 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, -1, -1, 0, -1],
    ]

    print("b1:", evaluate_board(b1, print_lists=True))
    print("b2:", evaluate_board(b2, print_lists=True))
    # print("b3:", evaluate_board(b3, print_lists=True))
    # print("b4:", evaluate_board(b4, print_lists=False))
    # print("b5:", evaluate_board(b5, print_lists=False))
    return


def test_best_move():
    b1 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, -1, -1, 0, -1],
    ]

    b1 = np.asarray(b1)
    env.reset(board=b1)

    ret = alpha_beta(env, 5, -INF, INF, True)
    print("want to place at 5, best is:", ret)


def test_win_next_turn():
    b1 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, -1, -1, 0, -1],
    ]
    print(" 1:", win_next_turn(b1, 1))
    print("-1:", win_next_turn(b1, -1))
    return


if __name__ == "__main__":
    # test_is_winning()
    test_eval()
    # test_best_move()
    # test_win_next_turn()
