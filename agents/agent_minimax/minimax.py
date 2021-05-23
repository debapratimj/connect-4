import pdb

import numpy as np
import math
from typing import Tuple, Optional
import numpy as np
from scipy.signal import convolve2d

from agents.common import BoardPiece, connected_four, check_end_state, SavedState, PlayerAction, initialize_game_state


def gen_moves(board: np.ndarray, player: np.int8) -> Tuple[list, list]:
    """
    Generate all possible moves for player at a particular game state by iterating through columns and searching for
    all possible moves.
    :param board:       np.ndarray
                        Current board represented by array for game state
    :param player:      BoardPiece
                        Current player taking the turn
    :return:            tuple
                        returns tuple containing list of states and their corresponding moves to get there from current
                        state
    """

    states = []
    col_ls = [3, 2, 4, 1, 5, 0, 6]
    moves = []
    for col in range(board.shape[1]):  # ite1rate over columns
        board_copy = board.copy()
        if np.int8(0) in board_copy[:, col]:  # free col for move
            board_copy[np.where(board_copy[:, col] == np.int8(0))[0][0], col] = player
            states += [board_copy]
            moves += [col]
    return states, moves


def opponent(player: BoardPiece) -> BoardPiece:
    """
    Returns opponent player to current given player
    :param player:      BoardPiece
                        Player for whom opponent is being calculated

    :return:            BoardPiece
                        opponent of current player
    """

    if player == BoardPiece(1):
        oppo = BoardPiece(2)
    else:
        oppo = BoardPiece(1)
    return oppo


def vanilla_minimax_gen_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
                             ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Apply vanilla minimax (no alpha beta pruning) and generate move for current player and returns player action
    :param board:           np.ndarray
                            Current state of the board
    :param player:          BoardPiece
                            Player for whom move is being generated
    :param saved_state:     Optional[SavedState]
                            optional saved state config of board
    :return:                tuple
                            tuple containing player action (move) and saved state
    """

    move, val = vanilla_minimax(board, player, 5, True)

    assert isinstance(move, int)

    return move, saved_state


def alpha_beta_minimax_gen_move(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
                                ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
        Apply minimax with alpha beta pruning and generate move for current player and returns player action
        :param board:           np.ndarray
                                Current state of the board
        :param player:          BoardPiece
                                Player for whom move is being generated
        :param saved_state:     Optional[SavedState]
                                optional saved state config of board
        :return:                tuple
                                tuple containing player action (move) and saved state
        """

    move, val = alpha_beta_minimax(board, player, -math.inf, math.inf, 5, True)

    assert isinstance(move, int)

    return move, saved_state


def vanilla_minimax(board: np.ndarray, player: BoardPiece, depth: int, maximizing: bool):
    """
    Vanilla minimax algorithm for playing connect 4. Returns move and heuristic value for current player via searching
    till a given depth into the game tree future
    :param board:       np.ndarray
                        Current state of the board represented by numpy array
    :param player:      BoardPiece
                        Player for whom move is being generated
    :param depth:       int
                        Integer representing how deep into the game tree is search by minimax agent for evaluating move
    :param maximizing:  bool
                        If True, then we maximise the score of the nodes (agent's move) if False, it is opponent's turn
                        who tries to minimise score
    :return:            Tuple
                        Returns tuple containing optimal move and heuristic value of move
    """

    if depth == 0 or check_end_state(board, player).name != 'STILL_PLAYING':
        return None, heuristic_minimax(board, player)
    if maximizing:
        val = -math.inf
        states_child, moves = gen_moves(board, player)
        move = np.random.choice(moves)
        for idx in range(len(states_child)):
            player_ = opponent(player)
            score = vanilla_minimax(states_child[idx], player_, depth - 1, False)[1]
            if score > val:
                val = score
                move = moves[idx]
        return move, val
    else:
        val = math.inf
        states_child, moves = gen_moves(board, player)
        move = np.random.choice(moves)
        for idx in range(len(states_child)):
            player_ = opponent(player)
            score = vanilla_minimax(states_child[idx], player_, depth - 1, True)[1]
            if score < val:
                val = score
                move = moves[idx]
        return move, val


def alpha_beta_minimax(board: np.ndarray, player: BoardPiece, alpha: float, beta: float, depth: int, maximizing: bool):
    """
        Vanilla minimax algorithm for playing connect 4. Returns move and heuristic value for current player via searching
        till a given depth into the game tree future
        :param board:       np.ndarray
                            Current state of the board represented by numpy array
        :param player:      BoardPiece
                            Player for whom move is being generated
        :param alpha:       float
                            alpha parameter to prune away computing node values whenever alpha > beta
        :param beta:        float
                            beta parameter to prune away computing node values whenever alpha > beta
        :param depth:       int
                            Integer representing how deep into the game tree is search by minimax agent for evaluating move
        :param maximizing:  bool
                            If True, then we maximise the score of the nodes (agent's move) if False, it is opponent's turn
                            who tries to minimise score
        :return:            Tuple
                            Returns tuple containing optimal move and heuristic value of move
        """

    if depth == 0 or check_end_state(board, player).name != 'STILL_PLAYING':
        return None, heuristic_minimax(board, player)
    if maximizing:
        val = -math.inf
        states_child, moves = gen_moves(board, player)
        move = np.random.choice(moves)
        for idx in range(len(states_child)):
            player_ = opponent(player)
            score = alpha_beta_minimax(states_child[idx], player_, alpha, beta, depth - 1, False)[1]
            if score > val:
                val = score
                move = moves[idx]
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return move, val
    else:
        val = math.inf
        states_child, moves = gen_moves(board, player)
        move = np.random.choice(moves)
        for idx in range(len(states_child)):
            player_ = opponent(player)
            score = alpha_beta_minimax(states_child[idx], player_, alpha, beta, depth - 1, True)[1]
            if score < val:
                val = score
                move = moves[idx]
            beta = min(beta, val)
            if alpha >= beta:
                break
        return move, val


def evaluate_window(window: list, player: BoardPiece) -> float:
    """
    Calculates score for heuristic minimax by counting pieces for agent and opponent player
    :param window:      list
                        List containing board snippets (windows) of length window_length as defined in heuristic function
                        [For connect 4, window_length is 4]
    :param player:      BoardPiece
                        Current player for whom board window is being calculated
    :return:            float
                        Returns float value of window for agent
    """

    score = 0

    if player == BoardPiece(1):
        opponent = BoardPiece(2)
    else:
        opponent = BoardPiece(1)

    if window.count(1) == 4:
        score += 100
    elif window.count(player) == 3 and window.count(BoardPiece(0)) == 1:
        score += 5
    elif window.count(player) == 2 and window.count(BoardPiece(0)) == 2:
        score += 2

    if window.count(opponent) == 3 and window.count(BoardPiece(0)) == 1:
        score -= 4

    return score


def heuristic_minimax(board: np.ndarray, player: BoardPiece) -> float:
    """
    Return heuristic board value of current player by generating different board snippets (windows) for e.g. center column
    diagonal, connect 4 in a row/column and calculating their value by calling evaluate_window.
    :param board:   np.ndarray
                    current board/game state
    :param player:  BoardPiece
                    Player for whom board value is being optimised
    :return:        float
                    scalar board value for current player.
    """

    value = 0
    window_length = 4
    cols = board.shape[1]
    rows = board.shape[0]

    # Center Column heuristic 
    center_array = [int(i) for i in list(board[:, cols // 2])]
    center_count = center_array.count(player)
    value += center_count * 3

    # Horizontal heuristic
    for r in range(rows):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(cols - 3):
            window = row_array[c:c + window_length]
            value += evaluate_window(window, player)

    # Vertical heuristic
    for c in range(cols):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(rows - 3):
            window = col_array[r:r + window_length]
            value += evaluate_window(window, player)

    # Board Diagonal heuristic
    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r + i][c + i] for i in range(window_length)]
            value += evaluate_window(window, player)

    for r in range(rows - 3):
        for c in range(cols - 3):
            window = [board[r + 3 - i][c + i] for i in range(window_length)]

            value += evaluate_window(window, player)

    return value


board_1 = np.random.choice(3, [6, 7])
board_config = initialize_game_state()
board_config[0, 0:4] = BoardPiece(1)
# print(heuristic_minimax(board_config, 1))

print(evaluate_window([1, 1, 1, 1], 1))
