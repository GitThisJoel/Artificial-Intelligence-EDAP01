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

env: ConnectFourEnv = gym.make("ConnectFour-v0")

# SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "http://lavender.blossom.dsek.se:3030/rooms/showdown/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = "nyckel"
STIL_ID = ["jo4383ba-s"]

INF = 10 ** 20
TOTAL_DEPTH = 4


def call_server(move):
    res = requests.post(
        SERVER_ADRESS + "move",
        data={
            "stil_id": STIL_ID,
            "move": move,  # -1 signals the system to start a new game. any running game is counted as a loss
            "api_key": API_KEY,
        },
    )
    # For safety some respose checking is done here
    if res.status_code != 200:
        print("Server gave a bad response, error code={}".format(res.status_code))
        exit()
    if not res.json()["status"]:
        print("Server returned a bad status. Return message: ")
        print(res.json()["msg"])
        exit()
    return res


def check_stats():
    res = requests.post(
        SERVER_ADRESS + "stats",
        data={
            "stil_id": STIL_ID,
            "api_key": API_KEY,
        },
    )

    stats = res.json()
    return stats


"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""


def opponents_move(env):
    env.change_player()  # change to oppoent
    avmoves = env.available_moves()
    if not avmoves:
        env.change_player()  # change back to student before returning
        return -1

    # TODO: Optional? change this to select actions with your policy too
    # that way you get way more interesting games, and you can see if starting
    # is enough to guarrantee a win

    action = random.choice(list(avmoves))
    # action = int(input("play: ")) - 1

    state, reward, done, _ = env.step(action)
    if done:
        if reward == 1:  # reward is always in current players view
            reward = -1
    env.change_player()  # change back to student before returning
    return state, reward, done


def evaluate_list(marker_list, length=4, print_lists=False):
    ones = marker_list.count(1)
    minus = marker_list.count(-1)
    zeros = length - ones - minus

    if print_lists and zeros != length and (ones > 1 or minus > 1):
        print(marker_list)

    # TODO: make this general for any length
    score = 0
    if ones == 4:
        score = 10000
    elif ones == 3 and zeros == 1:
        score = 10
    elif ones == 2 and zeros == 2:
        score = 5

    elif minus == 2 and zeros == 2:
        score = -6
    elif minus == 3 and zeros == 1:
        score = -11
    elif minus == 4:
        score = -10000 + 1

    return score


def evaluate_board(board, length=4):
    score = 0

    # rows
    for i in range(len(board)):
        row = board[i]
        for j in range(len(board[i]) - length + 1):
            squares = list(row[j : j + length])
            score += evaluate_list(squares)

    for i in range(len(board[0])):  # cols
        col = board[:, i]
        for j in range(len(board) - length + 1):
            squares = list(col[j : j + length])
            score += evaluate_list(squares)

    # diagonal
    for i in range(len(board) - length + 1):
        for j in range(len(board[i]) - length + 1):
            squares = [board[i + k][j + k] for k in range(length)]
            score += evaluate_list(squares)

    # other diagonal
    reverse_board = np.fliplr(board)
    for i in range(len(board) - length + 1):
        for j in range(len(board[i]) - length + 1):
            squares = [reverse_board[i + k][j + k] for k in range(length)]
            score += evaluate_list(squares)

    return score


def alpha_beta(curr_env, depth, alpha, beta, maximizing_player):
    moves = curr_env.available_moves()

    if depth == 0 or len(moves) == 0 or curr_env.is_win_state():  # or terminal node:
        return evaluate_board(curr_env.board)

    moves = list(moves)
    random.shuffle(moves)

    if maximizing_player:
        value = -INF
        for m in moves:
            new_env = copy.deepcopy(curr_env)
            new_env.change_player()
            new_env.step(m)

            # value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
            value = max(value, alpha_beta(new_env, depth - 1, alpha, beta, False))

            if value >= beta:
                break

            alpha = max(alpha, value)

            if depth > 1 and False:
                print(f"at {depth=}, {m=} and {value=}")

        return value  # - 2 * (TOTAL_DEPTH - depth))

    else:
        value = INF
        for m in moves:
            new_env = copy.deepcopy(curr_env)
            new_env.change_player()
            new_env.step(m)

            # value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
            value = min(value, alpha_beta(new_env, depth - 1, alpha, beta, True))

            if value <= alpha:
                break

            beta = min(beta, value)
            if depth > 1 and False:
                print(f"at {depth=}, {m=} and {value=}")
        return value  # value + 2 * (TOTAL_DEPTH - depth)


def student_move():
    """
    TODO: Implement your min-max alpha-beta pruning algorithm here.
    Give it whatever input arguments you think are necessary
    (and change where it is called).
    The function should return a move from 0-6
    """

    moves = env.available_moves()
    hi = -INF
    move_scores = [None] * 7
    for m in moves:
        new_env = copy.deepcopy(env)
        new_env.step(m)

        value = alpha_beta(new_env, TOTAL_DEPTH, -INF, INF, False)

        move_scores[m] = value

        if value > hi:
            best_move = m
            hi = value

    print(f"our {best_move=}, with {hi=}")
    print(f"{move_scores=}")

    return best_move


def play_game(vs_server=False):
    """
    The reward for a game is as follows. You get a
    botaction = random.choice(list(avmoves)) reward from the
    server after each move, but it is 0 while the game is running
    loss = -1
    win = +1
    draw = +0.5
    error = -10 (you get this if you try to play in a full column)
    Currently the player always makes the first move
    """

    # default state
    state = np.zeros((6, 7), dtype=int)

    # setup new game
    if vs_server:
        # Start a new game
        res = call_server(
            -1
        )  # -1 signals the system to start a new game. any running game is counted as a loss

        # This should tell you if you or the bot starts
        print(res.json()["msg"])
        # print(res.json())
        botmove = res.json()["botmove"]
        state = np.array(res.json()["state"])
        env.reset(board=state)
    else:
        # reset game to starting state
        env.reset(board=None)
        # determine first player
        student_gets_move = random.choice([True, False])
        if student_gets_move:
            print("You start!\n")
        else:
            print("Bot starts!\n")

    # Print current gamestate
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state, "\n")

    done = False
    while not done:
        # Select your move
        stmove = student_move()  # TODO: change input here

        # make both student and bot/server moves
        if vs_server:
            # Send your move to server and get response
            res = call_server(stmove)
            print(res.json()["msg"])

            # Extract response values
            result = res.json()["result"]
            botmove = res.json()["botmove"]
            state = np.array(res.json()["state"])
            env.reset(board=state)
        else:
            if student_gets_move:
                # Execute your move
                avmoves = env.available_moves()
                if stmove not in avmoves:
                    print("You tried to make an illegal move! You have lost the game.")
                    break
                state, result, done, _ = env.step(stmove)

            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like
            # env.render(mode="console")

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(env)

        # Check if the game is over
        if result != 0:
            done = True
            if not vs_server:
                print("Game over. ", end="")
            if result == 1:
                print("You won!")
            elif result == 0.5:
                print("It's a draw!")
            elif result == -1:
                print("You lost!")
            elif result == -10:
                print("You made an illegal move and have lost!")
            else:
                print(f"Unexpected result result={result}")
            if not vs_server:
                print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
        else:
            print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

        # Print current gamestate
        print(state)
        print()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--local", help="Play locally", action="store_true")
    group.add_argument(
        "-o", "--online", help="Play online vs server", action="store_true"
    )
    parser.add_argument(
        "-s", "--stats", help="Show your current online stats", action="store_true"
    )
    args = parser.parse_args()

    # Print usage info if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.local:
        play_game(vs_server=False)
    elif args.online:
        play_game(vs_server=True)

    if args.stats:
        stats = check_stats()
        print(json.dumps(stats, indent=4))

    # TODO: Run program with "--online" when you are ready to play against the server
    # the results of your games there will be logged
    # you can check your stats bu running the program with "--stats"


if __name__ == "__main__":
    main()
