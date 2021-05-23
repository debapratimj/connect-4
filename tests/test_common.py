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
    board_ = apply_player_action(board=initialize_game_state(), action=3, player=np.int8(np.random.choice([1,2])))
    #CHECK WHETHER THE CHANG IS WHAT IS EXPECTED

    board_config_1 = np.zeros([6,7])
    board_config_1[0,1] = np.int8(1)

    board_config_1 = apply_player_action(board=board_config_1, action=1, player=np.int8(0))

    # breakpoint()
    expected_board_1 = np.zeros([6,7], dtype=int)
    expected_board_1[0,1] = np.int8(1)
    expected_board_1[1,1] = np.int8(0)



    #print(board_)
    #print('Expected Board')
    # print(expected_board_1)
    # print('\n')
    # print('Board after player action')
    # print(board_config_1)

    assert isinstance(board_, np.ndarray)
    assert np.all(board_config_1 == expected_board_1)


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

    win_board_1 = np.zeros((6, 7))
    win_board_1[0, 0:4] = np.array([1, 1, 1, 1], dtype=np.int8)

    state_2 = check_end_state(board=win_board_1, player=np.int8(1))
    print(state_1.name)
    print(state_2.name)

    assert state_1.name == 'STILL_PLAYING'
    assert state_2.name == 'IS_WIN'




