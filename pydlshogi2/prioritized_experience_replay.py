import numpy as np

import pydlshogi2.SumTree
import pydlshogi2.make_priority

def prioritized_experience_replay(self):
        sumtree = SumTree(2**20)
        self.features.fill(0)
        for i, hcpe in enumerate(hcpevec):
            # set_hcp(hcp): hcp形式を指定して盤面を設定する
            self.board.set_hcp(hcpe['hcp'])
            # make_input_features(board, features): 入力特徴量を作成する
            make_input_features(self.board, self.features[i])
            # make_move_label(move, color): 移動を表すラベルを作成。引数move: 指し手の数値を受け取る。
            self.move_label[i] = make_move_label(
                hcpe['bestMove16'], self.board.turn)
            # make_result(game_result, color): 対局結果から価値ネットワークの出力ラベル(1, 0, 0.5)に変換する。
            self.result[i] = make_result(hcpe['gameResult'], self.board.turn)
            
            sumtree = SumTree(2**20)
            
            sumtree.add(make_priority(hcpe['eval'], hcpe['gameResult'], self.board.turn))
            
            
            
        if self.device.type == 'cpu':
            return (self.torch_features.clone(),
                    self.torch_move_label.clone(),
                    self.torch_result.clone(),
                    )
        # deviceに転送
        else:
            return (self.torch_features.to(self.device),
                    self.torch_move_label.to(self.device),
                    self.torch_result.to(self.device),
                    )
      
