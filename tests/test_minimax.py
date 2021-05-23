import math
from agents.agent_minimax.minimax import gen_moves, heuristic_minimax, opponent, vanilla_minimax, \
    alpha_beta_minimax_gen_move, alpha_beta_minimax, vanilla_minimax_gen_move
from agents.common import BoardPiece, initialize_game_state


def test_gen_moves():
    player = BoardPiece(0)

    assert isinstance(gen_moves(initialize_game_state(), player), tuple)
    assert isinstance(gen_moves(initialize_game_state(), player)[0], list)
    assert isinstance(gen_moves(initialize_game_state(), player)[1], list)
    assert len(gen_moves(initialize_game_state(), player)[1]) == initialize_game_state().shape[1]

    board_config = initialize_game_state()
    board_config[:, 0] = BoardPiece(1)

    assert len(gen_moves(board_config, player)[0]) == initialize_game_state().shape[1] - 1


def test_heuristic_minimax():
    player = BoardPiece(1)
    opponent = BoardPiece(2)
    assert heuristic_minimax(initialize_game_state(), player) == 0
    assert heuristic_minimax(initialize_game_state(), opponent) == 0

    board_config = initialize_game_state()
    board_config[0, 0:4] = player

    assert heuristic_minimax(board_config, player) == 110  # four score + center col score + two score + three score

    board_config = initialize_game_state()
    board_config[0, 0:3] = opponent
    assert heuristic_minimax(board_config, player) == -4  # negative score as opponent has 3 pieces in a row


def test_opponent():
    player_1 = BoardPiece(1)
    player_2 = BoardPiece(2)

    assert isinstance(opponent(player_1), BoardPiece)
    assert isinstance(opponent(player_2), BoardPiece)
    assert opponent(player_1) == player_2
    assert opponent(player_2) == player_1


def test_alpha_beta_minimax():
    board_0 = initialize_game_state()
    player_1 = BoardPiece(1)
    player_2 = BoardPiece(2)

    assert isinstance(alpha_beta_minimax(board_0, player_1, -math.inf, math.inf, 5, True), tuple)

    # Check for maximum depth condition being reached
    assert alpha_beta_minimax(board_0, player_1, -math.inf, math.inf, 0, True)[0] is None

    # Starting configuration should return 0 value for either player:
    assert alpha_beta_minimax(board_0, player_1, -math.inf, math.inf, 5, True)[1] == 0
    assert alpha_beta_minimax(board_0, player_2, -math.inf, math.inf, 5, True)[1] == 0


def test_vanilla_minimax():
    board_0 = initialize_game_state()
    player_1 = BoardPiece(1)
    player_2 = BoardPiece(2)

    assert isinstance(vanilla_minimax(board_0, player_1, 4, True), tuple)

    # Check for maximum depth condition being reached
    assert vanilla_minimax(board_0, player_1, 0, True)[0] is None

    # Starting configuration should return 0 value for either player:
    assert vanilla_minimax(board_0, player_1, 1, True)[1] == 0
    assert vanilla_minimax(board_0, player_2, 1, True)[1] == 0


def test_alpha_beta_minimax_gen_move():
    board_0 = initialize_game_state()
    player_1 = BoardPiece(1)
    player_2 = BoardPiece(2)

    output = alpha_beta_minimax_gen_move(board_0, player_1, None)  # Get output at one go to reduce test time
    assert isinstance(output, tuple)
    assert len(output) == 2

    board_config_wining = initialize_game_state()
    board_config_wining[0, 0:3] = player_1

    # Check for winning move

    assert alpha_beta_minimax_gen_move(board_config_wining, player_1, None)[0] == 3


def test_vanilla_minimax_gen_move():
    board_0 = initialize_game_state()
    player_1 = BoardPiece(1)
    player_2 = BoardPiece(2)

    output = vanilla_minimax_gen_move(board_0, player_1, None)  # Get output at one go to reduce test time
    assert isinstance(output, tuple)
    assert len(output) == 2

    board_config_wining = initialize_game_state()
    board_config_wining[0, 0:3] = player_1

    # Check for winning move

    assert vanilla_minimax_gen_move(board_config_wining, player_1, None)[0] == 3
