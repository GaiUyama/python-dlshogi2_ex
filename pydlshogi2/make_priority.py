from cshogi import *

# 対局結果から報酬を作成
def make_result(game_result, color):
    if color == BLACK:
        if game_result == BLACK_WIN:
            return 1000
        if game_result == WHITE_WIN:
            return -1000
    else:
        if game_result == BLACK_WIN:
            return -1000
        if game_result == WHITE_WIN:
            return 1000
    return 0
