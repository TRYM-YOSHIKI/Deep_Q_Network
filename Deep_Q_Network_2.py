# coding:utf-8
# 必要なライブラリのインポート
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from keras import backend as K
import tensorflow as tf

#ソケット通信を行う関数をインポート
import server_test_2 as S
import Client as C


# 損失関数の定義
# 損失関数にhuber関数を使用
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)


# Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=14, action_size=7, hidden_size=20):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, 14))
        targets = np.zeros((batch_size, 7))
        mini_batch = memory.sample(batch_size)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            if action_b > 1:
                action_b += -1
            action_b += -1

            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号

        self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


# Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)


# カートの状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
            if action >= 1:
                action += 1
            action += 1

        else:
            action = np.random.choice([1,3,4,5,6,7,8])  # ランダムに行動する

        return action


# メイン関数開始----------------------------------------------------
# 初期設定--------------------------------------------------------
DQN_MODE = 1    # 1がDQN、0がDDQNです

num_episodes = 500  # 総試行回数
max_number_of_steps = 999  # 1試行のstep数
goal_average_reward = 15  # この報酬を超えると学習終了
num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納
gamma = 0.99    # 割引係数
islearned = 0  # 学習が終わったフラグ
# ---
hidden_size = 20               # Q-networkの隠れ層のニューロンの数
learning_rate = 0.01         # Q-networkの学習係数
memory_size = 10000            # バッファーメモリの大きさ
batch_size = 32                # Q-networkを更新するバッチの大記載
# ---
Max_state = [20, 400, 400, 3.14, 20, 20, 400, 400, 560, 3.14, 400, 400, 560, 3.14]  # 各データの最大値

# Qネットワークとメモリ、Actorの生成--------------------------------------------------------
mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
# plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
memory = Memory(max_size=memory_size)
actor = Actor()

# メインルーチン--------------------------------------------------------
for episode in range(num_episodes):  # 試行数分繰り返す
    surplus = 0  # 試行の回数を合わせるための変数
    print("#####スタート#####")
    state = S.Receive_data()  # 初めのstateを取得する
    state = np.array([float(d) for i, d in enumerate(state) if i < 14])  # floatの行列に変換
    # データを正規化する
    state_nrm = [st/max_st for (st, max_st) in zip(state, Max_state)]  # 正規化したデータ
    # list型のstateを、1行14列の行列に変換
    state = np.reshape(state, [1, 14])
    state_nrm = np.reshape(state_nrm, [1, 14])
    print(state_nrm)  # 初期値を確認する為

    reward = 0  # 報酬
    done = False  # 試行を終えるか判定するフラグ
    episode_reward = 0
    sub_score = 0
    if state[0][8] < 0.039:
        reward = 1
        episode_reward += 1
    
    # targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
    targetQN.model.set_weights(mainQN.model.get_weights())

    for t in range(max_number_of_steps):  # 1試行のループ
        action = actor.get_action(state_nrm, episode, mainQN)   # 時刻tでの行動を決定する

        #コマンドを送信する
        while True:
            if C.Send_command(action):  # 行動a_tの実行
                break

        next_state = S.Receive_data()  # 行動a_tの実行による、s_{t+1}を計算する
        next_state = np.array([float(d) for i, d in enumerate(next_state) if i < 14])  # floatの行列に変換
        # データを正規化する
        next_state_nrm = [st/max_st for (st, max_st) in zip(next_state, Max_state)]  # 正規化したデータ
        # list型のstateを、1行14列の行列に変換
        next_state = np.reshape(next_state, [1, 14])
        next_state_nrm = np.reshape(next_state_nrm, [1, 14])

        # 行動a_tの実行による、_R{t}を計算する
        get_flag = 0  # 旗を取ったか
        if next_state[0][8] < 0.039:  # 報酬を設定し、与える
            reward = 1
            get_flag = 1
        elif next_state[0][1] < 0:  # 画面外に出たら負の報酬
            reward = -1
            done = True
        elif next_state[0][1] > 400:
            reward = -1
            done = True
        elif next_state[0][2] < 0:
            reward = -1
            done = True
        elif next_state[0][2] > 400:
            reward = -1
            done = True
        elif state[0][8] - next_state[0][8] > 0:  # 旗に近づいていたら正の報酬
            reward = 0.4
        else:
            reward = 0
        
        if t == 998:  # 試行が最後だったら終える
            done = True

        if done:
            next_state_nrm = np.zeros(state_nrm.shape)  # 次の状態s_{t+1}はない

        episode_reward += get_flag  # 合計報酬を更新
        sub_score += reward

        # メモリの更新する
        memory.add((state_nrm, action, reward, next_state_nrm))
        state = next_state  # 状態更新
        state_nrm = next_state_nrm  # 状態更新


        # Qネットワークの重みを学習・更新する replay
        if (memory.len() > batch_size) and not islearned:
            mainQN.replay(memory, batch_size, gamma, targetQN)
            #print('####replay####')

        if DQN_MODE:
        # targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする
            targetQN.model.set_weights(mainQN.model.get_weights())

        if t % 10 == 0:
            print('######{}#######'.format(t))

        if done:
            total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # 報酬を記録
            print('{}回目 SCORE ({}step): {}, sub_score: {}'.format(episode, t, episode_reward, sub_score))
            surplus = max_number_of_steps - t - 1
            break

    #回数を合わせる
    action = np.random.choice([1,3,4,5,6,7,8])  # ランダムに行動する
    #コマンドを送信する
    while True:
        if C.Send_command(action):  # 行動a_tの実行
            break
    for k in range(surplus):
        state = S.Receive_data()  # 初めのstateを取得する
        action = np.random.choice([1,3,4,5,6,7,8])  # ランダムに行動する
        #コマンドを送信する
        while True:
            if C.Send_command(action):  # 行動a_tの実行
                break

    # 複数施行の平均報酬で終了を判断
    if total_reward_vec.mean() >= goal_average_reward:
        print('Episode %d train agent successfuly!' % episode)
        islearned = 0  # 学習済みフラグを更新
        try:
            targetQN.model.save('learned_ave15.h5')
        except:
            pass
        #break

    # 途中段階のモデルを保存する
    if episode == 50:
        targetQN.model.save('learned_episode50.h5')
    if episode == 30:
        targetQN.model.save('learned_episode30.h5')

targetQN.model.save('learned_episode100.h5')
        