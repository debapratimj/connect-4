import numpy as np
from agents.common import BoardPiece, NO_PLAYER, pretty_print_board, initialize_game_state, string_to_board, \
    connected_four, apply_player_action, check_end_state


def test_initialize_game_state():
    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    board = initialize_game_state()

    pp_string = pretty_print_board(board)

    assert isinstance(pp_string, str)
    print(pp_string)
    return pp_string



def test_string_to_board():
    string = test_pretty_print_board()

    board_ = string_to_board(string)

    assert isinstance(board_, np.ndarray)



def test_apply_player_action():
    board_ = apply_player_action(board=initialize_game_state(), action=3, player=np.int8(np.random.choice(1)))
    #CHECK WHETHER THE CHANG IS WHAT IS EXPECTED
    assert isinstance(board_, np.ndarray)


def test_connected_four():
    win = connected_four(board=initialize_game_state(), player=np.int8(np.random.choice([1, 2])))

    win_board_1 = np.zeros((6,7))
    win_board_1[0, 0:4] = np.array([1,1,1,1], dtype=np.int8)

    win_player_1 = connected_four(board = win_board_1 , player = np.int8(1))
    win_player_2 = connected_four(board = win_board_1 , player = np.int8(2))

    assert win is False
    assert win_player_1 is True
    assert win_player_2 is False

def test_check_end_state():
    state_1 = check_end_state(board=initialize_game_state(), player=np.int8(np.random.choice([1, 2])))

    print(state_1.name)