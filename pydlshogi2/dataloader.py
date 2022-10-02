import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import torch
import random

from cshogi import Board, HuffmanCodedPosAndEval
from pydlshogi2.features import FEATURES_NUM, make_input_features, make_move_label, make_result
from pydlshogi2.SumTree import SumTree
from pydlshogi2.make_priority import make_priority


class HcpeDataLoader:
    def __init__(self, files, batch_size, device, shuffle=False, per=False):
        self.load(files)
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        # PERを使用するか
        self.per = per

        # torch.empty: 初期化されていないデータで満たされたテンソルを返す
        # torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False)
        # pin_memory（bool 、optional）–設定されている場合、返されたテンソルは固定されたメモリに割り当てられる。CPUテンソルでのみ機能する。
        self.torch_features = torch.empty((batch_size, FEATURES_NUM, 9, 9), dtype=torch.float32, pin_memory=True)
        self.torch_move_label = torch.empty((batch_size), dtype=torch.int64, pin_memory=True)
        self.torch_result = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)
        
        # 優先度
        self.torch_priority = torch.empty((batch_size, 1), dtype=torch.float32, pin_memory=True)

        # インスタンス化
        self.features = self.torch_features.numpy()
        self.move_label = self.torch_move_label.numpy()
        # reshape(-1): 行ベクトルに変換
        self.result = self.torch_result.numpy().reshape(-1)
        
        # 優先度
        self.priority = self.torch_priority.numpy().reshape(-1)

        self.i = 0
        # ThreadPoolExecutor: マルチスレッドによる並列化を行う。コンストラクタ引数 max_workers でワーカー、すなわちスレッドの最大数を指定する。
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.board = Board()
        
        '''
        if self.per:
            self.per_sort()
        '''

    def load(self, files):
        data = []
        if type(files) not in [list, tuple]:
            files = [files]
        for path in files:
            # ファイルが存在するかチェック
            if os.path.exists(path):
                logging.info(path)
                # append: リストの末尾にデータを追加
                # fromfile: ファイルの名前と配列のデータ型を入力パラメーターとして受け取り、配列を返す
                data.append(np.fromfile(path, dtype=HuffmanCodedPosAndEval))
            else:
                logging.warn('{} not found, skipping'.format(path))
        # concatenate(): 複数のNumPy配列ndarrayを結合（連結）する。結合する軸はデフォルト0で縦
        self.data = np.concatenate(data)
        
        # print(self.data)
        
            
    def per_sort(self, hcpevec):
        hcpes = []
        n_data = []
        p_data = []
        
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe['hcp'])
            self.priority[i] = make_priority(hcpe['eval'], hcpe['gameResult'], self.board.turn)
            hcpes.extend(hcpe)
            # hcpes.append(hcpe)
            # hcpes[i] = hcpe
            
            if self.priority[i] == 0:
                n_data.extend(hcpe)
            else:
                p_data.extend(hcpe)
        
        np.random.shuffle(n_data)
        np.random.shuffle(p_data)
        hcpevec = n_data.extend(p_data)
        
        '''
        priority_sort = sorted(self.priority)
        
        n_data = []  #priority = 0 のデータ
        p_data = []  #priority = 1 のデータ
        i = 0
        j = 0
        
        for k in range(len(hcpevec)):
            if self.priority[k] == 0:
                # n_data.extend(hcpes[k])
                # n_data.append(hcpes[k])
                n_data[i] = hcpes[k]
                i += 1
            else:
                # p_data.extend(hcpes[k])
                # p_data.append(hcpes[k])
                p_data[j] = hcpes[k]
                j += 1
        # random.shuffle()?        
        np.random.shuffle(n_data)
        np.random.shuffle(p_data)
        
        hcpevec = n_data.extend(p_data)
        '''
        '''
        for i in range(len(self.data)):
            for j in range(len(self.data)):
                if priority_sort[i] == self.priority[j]:
                    self.data[i] = hcpes[j]
                    break
        '''
        

    # ミニバッチ作成
    def mini_batch(self, hcpevec):
        # 入力特徴量を0に初期化
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

        
    # 優先度付きミニバッチ
    def per_mini_batch(self, hcpevec):
        '''
        sumtree = SumTree(self.batch_size)
        # SumTree作成
        for i, hcpe in enumerate(hcpevec):
            # 優先度
            self.priority[i] = make_priority(hcpe['eval'], hcpe['gameResult'], self.board.turn)
            sumtree.add(self.priority[i], hcpe)
            
        # ミニバッチ作り直し
        for i in range(self.batch_size):
            s = random.uniform(0, sumtree.total())
            idx, p, data = sumtree.get(s)
            hcpevec[i] = data
        '''    
        
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe['hcp'])
            # 優先度
            self.priority[i] = make_priority(hcpe['eval'], hcpe['gameResult'], self.board.turn)
            print(self.priority[i])
        
        hcpevec = np.random.choice(hcpevec, self.batch_size, self.priority, replace=False)
        
        print(1)
        
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
        
    
    
    def sample(self):
        # np.random.choice(a, size, replace=False, p): 配列やリストからランダムに要素を取り出す。
        # size: 出力する配列のshapeを指定
        # replace=False: 重複なし
        # 配列中の各要素の生起確率をオプションpを与えることで設定可能
        return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))
    
    
    '''
    def PER_sample(self):
        # np.random.choice(a, size, replace=False, p): 配列やリストからランダムに要素を取り出す。
        # size: 出力する配列のshapeを指定
        # replace=False: 重複なし
        # 配列中の各要素の生起確率をオプションpを与えることで設定可能
        # return self.mini_batch(np.random.choice(self.data, self.batch_size, replace=False))
    
        s = random.uniform(0, sumtree.total())
        
        sumtree.get(s)
    '''
    
    # デバッグ
    def debug(self):
        hcpevec = self.data[self.i:self.i+self.batch_size]
        '''
        for i, hcpe in enumerate(hcpevec):
            self.board.set_hcp(hcpe['hcp'])
            # 優先度
            self.priority[i] = make_priority(hcpe['eval'], hcpe['gameResult'], self.board.turn)
            # print(self.priority[i])
        '''
        self.per_sort(hcpevec)
        #hcpevec = np.random.choice(hcpevec, self.batch_size, self.priority, replace=False)
        
        for i, hcpe in enumerate(hcpevec):
          print(hcpe)

        return hcpevec
    
        # return self.priority
    

    def pre_fetch(self):
        hcpevec = self.data[self.i:self.i+self.batch_size]
        self.i += self.batch_size
        if len(hcpevec) < self.batch_size:
            return
        
        # per=True
        if self.per:
            self.per_sort(hcpevec)
        
        self.f = self.executor.submit(self.mini_batch, hcpevec)
        
        '''
        # per=True
        if self.per:
            self.f = self.executor.submit(self.per_mini_batch, hcpevec)
        # per=False
        else:
            # executer.submit(task, i): 並列タスクを実行するメソッド
            self.f = self.executor.submit(self.mini_batch, hcpevec)
        '''
          

    def __len__(self):
        return len(self.data)

    # オブジェクトがイテレータとして振る舞うには、そのオブジェクトに __next__() と __iter__() という二つの特殊メソッドが必要になる
    # forを始める前に__iter__メソッドが実行される
    def __iter__(self):
        self.i = 0
        if self.shuffle:
            # np.random.shuffle(): 受け取った配列の要素をシャッフルして並び替える
            np.random.shuffle(self.data)
        
        self.pre_fetch()
        return self

    # forブロックのループのたびに__next__メソッドが実行される
    def __next__(self):
        # 要素を全て取り出しきっているかチェック
        if self.i > len(self.data):
            # 要素を全て取り出しきると例外StopIteration()
            raise StopIteration()

        result = self.f.result()
        self.pre_fetch()

        return result
