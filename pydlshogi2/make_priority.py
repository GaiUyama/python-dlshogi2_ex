import math
import numpy as np

from cshogi import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_priority(eval, game_result, color):
    if color == BLACK:
        if game_result == BLACK_WIN:
            return math.fabs(1000-eval)
        if game_result == WHITE_WIN:
            return math.fabs(-1000-eval)
    else:
        if game_result == BLACK_WIN:
            return math.fabs(-1000-eval)
        if game_result == WHITE_WIN:
            return math.fabs(1000-eval)
    return math.fabs(eval)

def r_make_priority():
