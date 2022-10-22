import argparse
from cshogi import HuffmanCodedPosAndEval, Board, BLACK, move16
from cshogi import CSA
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir')
parser.add_argument('hcpe_train')
parser.add_argument('hcpe_test')
parser.add_argument('--filter_moves', type=int, default=50)
parser.add_argument('--filter_rating', type=int, default=3500)
parser.add_argument('--test_ratio', type=float, default=0.1)
args = parser.parse_args()

POSNUM_BORDER = 1
TRAIN_NUM = 2400000
TEST_NUM = TRAIN_NUM * 0.11111

# glob.glob(args.csa_dir/**/*.csa) 
# ファイルパスをリストとして取得
# **: 同一階層以外のファイルを探す, recursive=True: ファイルを再帰的に探す
csa_file_list = glob.glob(os.path.join(args.csa_dir, '**', '*.csa'), recursive=True)

# ファイルリストをシャッフル
file_list_train, file_list_test = train_test_split(csa_file_list, test_size=args.test_ratio)

hcpes = np.zeros(1024, HuffmanCodedPosAndEval)

# hcpe_train = train.hcpe, hcpe_test = test.hcpe
# 'wb': バイナリファイルを書き込みモードで開く
f_train = open(args.hcpe_train, 'wb')
f_test = open(args.hcpe_test, 'wb')

flag = 0

board = Board()
# zip(): file_list = [file_list_train, file_list_test], f = [f_train, f_test]
for file_list, f in zip([file_list_train, file_list_test], [f_train, f_test]):
    kif_num = 0
    position_num = 0
    for filepath in file_list:
        # CSA.Parser.parse_file(): CSA形式の棋譜の読み込み
        for kif in CSA.Parser.parse_file(filepath):
            # 投了、千日手、宣言勝ちで終了した棋譜以外を除外
            if kif.endgame not in ('%TORYO', '%SENNICHITE', '%KACHI'):
                # continue: ループの最初に戻ってやり直す
                continue
            # 手数が少ない棋譜を除外
            if len(kif.moves) < args.filter_moves:
                continue
            # レーティングの低いエンジンの対局を除外
            if args.filter_rating > 0 and min(kif.ratings) < args.filter_rating:
                continue

            # 開始局面を設定
            board.set_sfen(kif.sfen)
            p = 0
            # try: 例外が発生するかもしれないが、実行したい処理
            try:
                # enumerate() : 要素のインデックスと要素を同時に取り出す
                # moves: 開始局面からの指し手のリスト
                # scores: 開始局面からの評価値のリスト
                # comments: 開始局面からの指し手のコメントのリスト
                for i, (move, score, comment) in enumerate(zip(kif.moves, kif.scores, kif.comments)):
                    # 局面数固定
                    if POSNUM_BORDER == 1:
                      total = position_num + p
                      if flag <= TRAIN_NUM:
                        if total == TRAIN_NUM:
                          break
                      else:
                        if total == TEST_NUM:
                          break
                      flag += 1
                                    
                    # 訓練局面数固定
                    '''
                    total = position_num + p
                    if total == 2400000:
                      break
                    '''
                    
                    # 不正な指し手のある棋譜を除外
                    # is_legal(move): moveが合法手かチェック。真偽値を返す。
                    if not board.is_legal(move):
                        # raise: 例外を発生させる (exceptに移動)
                        raise Exception()
                    hcpe = hcpes[p]
                    p += 1
                    # 局面はhcpに変換
                    # to_hcp(hcp): 現局面を表すhcp形式のndarrayを取得する。引数には、結果を受け取るndarrayを指定する。
                    board.to_hcp(hcpe['hcp'])
                    # 16bitに収まるようにクリッピングする
                    eval = min(32767, max(score, -32767))
                    # 手番側の評価値にする
                    hcpe['eval'] = eval if board.turn == BLACK else -eval
                    # 指し手の32bit数値を16bitに切り捨てる
                    # move16(move): 32bitのmoveを16bitのmoveに変換する
                    hcpe['bestMove16'] = move16(move)
                    # 勝敗結果
                    hcpe['gameResult'] = kif.win
                    # push(move): 指し手moveを盤面に適用する
                    board.push(move)
            # except: 例外発生：除外対象の棋譜であるとき
            except:
                print(f'skip {filepath}')
                # 処理をスキップ
                continue

            if p == 0:
                continue

            # tofile(): NumPy配列(hcpes[:p])をバイナリファイル(f)に保存
            # hcpe_train = train.hcpe, hcpe_test = test.hcpe
            # f_train = open(args.hcpe_train, 'wb'), f_test = open(args.hcpe_test, 'wb')
            # f = [f_train, f_test]
            hcpes[:p].tofile(f)

            # 棋譜数
            kif_num += 1
            # 局面数
            position_num += p

    print('kif_num', kif_num)
    print('position_num', position_num)
