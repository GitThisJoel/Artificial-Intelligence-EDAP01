import random
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, unique
from operator import itemgetter
from typing import Tuple, NamedTuple, Hashable, Optional

import gym
import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from gym import error
from gym import spaces

from gym_connect_four.envs.render import render_board

@unique
class ResultType(Enum):
    NONE = None
    DRAW = 0
    WIN1 = 1
    WIN2 = -1

    def __eq__(self, other):
        """
        Need to implement this due to an unfixed bug in Python since 2017: https://bugs.python.org/issue30545
        """
        return self.value == other.value


class ConnectFourEnv(gym.Env):
    """
    Description:
        ConnectFour game environment

    Observation:
        Type: Discreet(6,7)

    Actions:
        Type: Discreet(7)
        Num     Action
        x       Column in which to insert next token (0-6)

    Reward:
        Reward is 0 for every step.
        If there are no other further steps possible, Reward is 0.5 and termination will occur
        If it's a win condition, Reward will be 1 and termination will occur
        If it is an invalid move, Reward will be -1 and termination will occur

    Starting State:
        All observations are assigned a value of 0

    Episode Termination:
        No more spaces left for pieces
        4 pieces are present in a line: horizontal, vertical or diagonally
        An attempt is made to place a piece in an invalid location
    """

    metadata = {'render.modes': ['human']}

    LOSS_REWARD = -1
    DEF_REWARD = 0
    DRAW_REWARD = 0.5
    WIN_REWARD = 1

    class StepResult(NamedTuple):

        res_type: ResultType

        def get_reward(self, player: int):
            if self.res_type is ResultType.NONE:
                return ConnectFourEnv.DEF_REWARD
            elif self.res_type is ResultType.DRAW:
                return ConnectFourEnv.DRAW_REWARD
            else:
                return {ResultType.WIN1.value: ConnectFourEnv.WIN_REWARD, ResultType.WIN2.value: ConnectFourEnv.LOSS_REWARD}[
                    self.res_type.value * player]

        def is_done(self):
            return self.res_type != ResultType.NONE

    def __init__(self, board_shape=(6, 7), window_width=512, window_height=512):
        super(ConnectFourEnv, self).__init__()

        self.board_shape = board_shape

        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=board_shape,
                                            dtype=int)
        self.action_space = spaces.Discrete(board_shape[1])

        self.__current_player = 1
        self.__board = np.zeros(self.board_shape, dtype=int)

        self.__player_color = 1
        self.__screen = None
        self.__window_width = window_width
        self.__window_height = window_height
        self.__rendered_board = self._update_board_render()

    def change_player(self):
        self.__current_player *= -1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        step_result = self._step(action)
        reward = step_result.get_reward(self.__current_player)
        done = step_result.is_done()
        return self.__board.copy(), reward, done, {}

    def _step(self, action: int) -> StepResult:
        result = ResultType.NONE

        if not self.is_valid_action(action):
            raise Exception(
                'Unable to determine a valid move! Maybe invoke at the wrong time?'
            )

        # Check and perform action
        for index in list(reversed(range(self.board_shape[0]))):
            if self.__board[index][action] == 0:
                self.__board[index][action] = self.__current_player
                break

        # Check if board is completely filled
        if np.count_nonzero(self.__board[0]) == self.board_shape[1]:
            result = ResultType.DRAW
        else:
            # Check win condition
            if self.is_win_state():
                result = ResultType.WIN1 if self.__current_player == 1 else ResultType.WIN2
        return self.StepResult(result)

    @property
    def board(self):
        return self.__board.copy()

    def reset(self, board: Optional[np.ndarray] = None) -> np.ndarray:
        self.__current_player = 1
        if board is None:
            self.__board = np.zeros(self.board_shape, dtype=int)
        else:
            self.__board = board
        self.__rendered_board = self._update_board_render()
        return self.board

    def render(self, mode: str = 'console', close: bool = False) -> None:
        if mode == 'console':
            replacements = {
                self.__player_color: 'A',
                0: ' ',
                -1 * self.__player_color: 'B'
            }

            def render_line(line):
                return "|" + "|".join(
                    ["{:>2} ".format(replacements[x]) for x in line]) + "|"

            hline = '|---+---+---+---+---+---+---|'
            print(hline)
            for line in np.apply_along_axis(render_line,
                                            axis=1,
                                            arr=self.__board):
                print(line)
            print(hline)

        elif mode == 'human':
            if self.__screen is None:
                pygame.init()
                self.__screen = pygame.display.set_mode(
                    (round(self.__window_width), round(self.__window_height)))

            if close:
                pygame.quit()

            self.__rendered_board = self._update_board_render()
            frame = self.__rendered_board
            surface = pygame.surfarray.make_surface(frame)
            surface = pygame.transform.rotate(surface, 90)
            self.__screen.blit(surface, (0, 0))

            pygame.display.update()
        else:
            raise error.UnsupportedMode()

    def close(self) -> None:
        pygame.quit()

    def is_valid_action(self, action: int) -> bool:
        return self.__board[0][action] == 0

    def _update_board_render(self) -> np.ndarray:
        return render_board(self.__board,
                            image_width=self.__window_width,
                            image_height=self.__window_height)

    def is_win_state(self) -> bool:
        # Test rows
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1] - 3):
                value = sum(self.__board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*self.__board)]
        for i in range(self.board_shape[1]):
            for j in range(self.board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += self.__board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        reversed_board = np.fliplr(self.__board)
        # Test reverse diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        return False

    def available_moves(self) -> frozenset:
        return frozenset(
            (i for i in range(self.board_shape[1]) if self.is_valid_action(i)))
