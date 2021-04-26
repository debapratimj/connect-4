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

def printBoard (board):
    """Print the board."""
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



boardx = np.full(((6,7)) , 1, dtype = np.int8 )
boardx = np.random.randint(3 , size= [6,7] , dtype = np.int8 )
#print(boardx)
pp_string = printBoard(boardx)
print('fff')

print(pp_string)
ls =[]

string = 'a '
ls =[ 0 if i == ' ' else 1 if i == 'X' else 2 for i in pp_string]

#ls  = [x for x in string ]
str_to_board = np.flip(np.array(ls ).reshape(6,7) , 0)
print(len(ls))


print(len(pp_string))

print(str_to_board)


print(boardx)
print(boardx.shape)
board = boardx
action  = 4

row_index = np.argmin(board[:,action] == 0 )

board[row_index][action] = np.int8(2)

print(board)


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    win = False
    winner_piece = None
    if last_action != None:
        pass
    else:
        for c in range(board.shape[1]):
            for r in range(board.shape[0]):
                if c - 3 >= 0:
                    # check rows to left
                    if board[r][c] == player  and board[r][c] == board[r][c-1] == board[r][c-2] == board[r][c-3]:
                        winner_piece = BoardPiece(board[r][c])
                        win = True
                        return win  , [r,c]
                    # check diagonals going bottom left
                    if r <= 2 :
                        if board[r][c]== player and board[r][c] == board[r+1][c-1] == board[r+2][c-2] == board[r+3][c-3] :
                            win = True
                            winner_piece = BoardPiece(board[r][c])
                            return win , [r,c]
                else:
                    # checks diagonals going bottom right
                    if  r <= 2 :
                        if board[r][c] == player  and board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3] :
                            win = True
                            winner_piece = BoardPiece(board[r][c])
                            return win , [r,c]

                if r - 3 >= 0 :
                    # checks columns to the top
                    if board[r][c] == player  and board[r][c] == board[r-1][c] == board[r-2][c] == board[r-3][c] :
                        winner_piece = BoardPiece(board[r][c])
                        win = True
                        return win , [r,c]


    return win ,  [r,c]

    raise NotImplementedError()


win  , [r,c]  = connected_four(board, np.int8(2) , None)

#print(win , [r,c] )

print(5 in board)

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

