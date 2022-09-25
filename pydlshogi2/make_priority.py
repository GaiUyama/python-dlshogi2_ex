import math
import numpy as np

from cshogi import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_priority(eval, game_result, color):
    if color == BLACK:
        if game_result == BLACK_WIN:  #先手勝ち
            #return math.fabs(100000-eval)
            return 0
        else:  #先手負け
            return 1
        '''
        if game_result == WHITE_WIN:
            return math.fabs(-100000-eval)
            '''
        
    else:
        '''
        if game_result == BLACK_WIN:
            return math.fabs(-100000-eval)
        else:
            return 0
        '''
        if game_result == WHITE_WIN:  #後手勝ち
            #return math.fabs(100000-eval)
            return 0
        else:  #後手負け
            return 1
            
    return 0 # math.fabs(eval)

# def r_make_priority():
