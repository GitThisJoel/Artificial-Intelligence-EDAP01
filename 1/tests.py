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


def evaluate_list(marker_list, length=4):
    ones = marker_list.count(1)
    minus = marker_list.count(-1)
    zeros = length - ones - minus

    score = 0
    if ones == 4:
        score += 100000
    elif ones == 3 and zeros == 1:
        score += 5000
    elif ones == 2 and zeros == 2:
        score += 500

    elif minus == 2 and zeros == 2:
        score -= 600
    elif minus == 3 and zeros == 1:
        score -= 5100
    elif minus == 4:
        score -= 100000 - 1

    return score


def evaluate_board(board, length=4):  # maybe not optimal max score
    score = 0
    l = [0] * 4

    # row
    for i in range(len(board)):
        for j in range(len(board[i]) - length + 1):
            score += evaluate_list(board[i][j : j + length])

    for i in range(len(board) - length + 1):
        # down
        for j in range(len(board[i])):
            for n in range(0, length):
                l[n] = board[i + n][j]
            score += evaluate_list(l)

        # diag
        for j in range(len(board[i]) - length + 1):
            for n in range(length):
                l[n] = board[i + n][j + n]
            score += evaluate_list(l)

        for j in range(length - 1, len(board[i])):
            for n in range(length):
                l[n] = board[i - n][j - n]
            score += evaluate_list(l)
    return score


def test_eval():
    b1 = [
        [0, -1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, -1, 1, 0, 0],
        [0, 1, -1, 1, 0],
        [0, -1, 1, -1, 0],
    ]
    print(evaluate_board(b1))
    return


if __name__ == "__main__":
    # test_is_winning()
    test_eval()
