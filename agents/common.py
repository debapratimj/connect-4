from enum import Enum
from typing import Optional
import numpy as np

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiece_Print = str  # dtype for string representation of BoardPiece
NO_PLAYER_Print = str(' ')
PLAYER1_Print = str('X')
PLAYER2_Print = str('O')

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    init_state = np.zeros([6, 7], dtype=BoardPiece(0))

    return init_state

    # raise NotImplementedError('Didnt initialise game state')


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    board = np.flip(board, 0)  # flip vertically to get board[x][y] in lower left
    print('|==============|')
    pp_string = str('')
    for y in range(6):
        pp_string = pp_string + ''.join(
            NO_PLAYER_Print if board[y][x] == NO_PLAYER else PLAYER1_Print if board[y][x] == PLAYER1 else PLAYER2_Print
            for x in range(7))

        print(str('| ') + ' '.join(
            NO_PLAYER_Print if board[y][x] == NO_PLAYER else PLAYER1_Print if board[y][x] == PLAYER1 else PLAYER2_Print
            for x in range(7)) + str(' |'))
    print('|==============|')
    print(str('| ') + ' '.join(map(str, range(7))) + str(' |'))
    print()

    return pp_string

    # raise NotImplementedError()


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    ls = [0 if i == ' ' else 1 if i == 'X' else 2 for i in pp_board]

    str_to_board = np.flip(np.array(ls).reshape(6, 7), 0)

    return str_to_board

    # raise NotImplementedError()


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """

    if copy:
        board = np.copy(board)
    else:
        pass
    row_index = np.argmin(board[:, action]) == 0

    board[row_index,action] = player

    # Throw error is the move is not valid?

    return board
    # raise NotImplementedError()


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    :rtype: object
    """
    win = False
    if last_action is not None:
        pass
        # optimised code
    else:
        for c in range(board.shape[1]):
            for r in range(board.shape[0]):
                if c - 3 >= 0:
                    # check rows to left
                    if board[r][c] == player and board[r][c] == board[r][c - 1] == board[r][c - 2] == board[r][c - 3]:
                        win = True
                        return win
                    # check diagonals going bottom left
                    if r <= 2:
                        if board[r][c] == player and board[r][c] == board[r + 1][c - 1] == board[r + 2][c - 2] == \
                                board[r + 3][c - 3]:
                            win = True
                            return win
                else:
                    # checks diagonals going bottom right
                    if r <= 2:
                        if board[r][c] == player and board[r][c] == board[r + 1][c + 1] == board[r + 2][c + 2] == \
                                board[r + 3][c + 3]:
                            win = True
                            return win

                if r - 3 >= 0:
                    # checks columns to the top
                    if board[r][c] == player and board[r][c] == board[r - 1][c] == board[r - 2][c] == board[r - 3][c]:
                        win = True
                        return win

    return win

    # raise NotImplementedError()


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    win = connected_four(board, player)

    if win:
        state = GameState.IS_WIN

    else:
        if NO_PLAYER in board:
            state = GameState.STILL_PLAYING
        else:
            state = GameState.IS_DRAW
    return state
    # raise NotImplementedError()
