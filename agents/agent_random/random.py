import numpy as np
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from typing import Callable, Tuple, Optional


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    contradiction = True
    while contradiction:
        rand_col = np.random.choice(board.shape[1])
        if np.int8(0) in board[:, rand_col]:
            contradiction = False
            action = rand_col
    return action, saved_state
