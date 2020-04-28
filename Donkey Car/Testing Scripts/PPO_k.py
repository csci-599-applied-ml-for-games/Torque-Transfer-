import numpy as np

import gym

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import numba as nb
from tensorboardX import SummaryWriter
from gym_torcs import TorcsEnv

ENV = 'TORCS'
CONTINUOUS = True

EPISODES = 100000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 1.0 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 128
BATCH_SIZE = 256
NUM_ACTIONS = 1
NUM_STATE = 8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss


class Agent:
    def __init__(self):
        self.critic = self.build_critic()
        if CONTINUOUS is False:
            self.actor = self.build_actor()
        else:
            self.actor = self.build_actor_continuous()

        self.env = env = TorcsEnv(vision=True, throttle=False, gear_change=False)
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.name = self.get_name()
        self.writer = SummaryWriter(self.name)
        self.gradient_steps = 0

    def get_name(self):
        name = 'AllRuns/'
        if CONTINUOUS is True:
            name += 'continous/'
        else:
            name += 'discrete/'
        name += ENV
        return name

    def get_cnn(self):
        Input_1 = Input(shape=(64, 64, 3), name='Input_1')
        Convolution2D_1 = Conv2D(4, kernel_size=3, padding='same', activation='relu')(Input_1)
        Convolution2D_2 = Conv2D(4, kernel_size=3, padding='same', activation='relu')(Convolution2D_1)
        # Convolution2D_2 = BatchNormalization()(Convolution2D_2)
        MaxPooling2D_1 = MaxPooling2D()(Convolution2D_2)
    
        Convolution2D_5 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_1)
        Convolution2D_6 = Conv2D(8, kernel_size=3, padding='same', activation='relu')(Convolution2D_5)
        # Convolution2D_6 = BatchNormalization()(Convolution2D_6)
        MaxPooling2D_2 = MaxPooling2D()(Convolution2D_6)
    
        Convolution2D_7 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_2)
        Convolution2D_8 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_7)
        Convolution2D_11 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_8)
        # Convolution2D_11 = BatchNormalization()(Convolution2D_11)
        MaxPooling2D_3 = MaxPooling2D()(Convolution2D_11)
    
        Convolution2D_9 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_3)
        Convolution2D_10 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(Convolution2D_9)
        Convolution2D_12 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_10)
        # Convolution2D_12 = BatchNormalization()(Convolution2D_12)
        MaxPooling2D_4 = MaxPooling2D(name='MaxPooling2D_4')(Convolution2D_12)
    
        Convolution2D_13 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(MaxPooling2D_4)
        Convolution2D_14 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(Convolution2D_13)
        Convolution2D_16 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(Convolution2D_14)
        # Convolution2D_16 = BatchNormalization()(Convolution2D_16)
        MaxPooling2D_5 = MaxPooling2D(name='MaxPooling2D_5')(Convolution2D_16)
    
        Flatten_1 = Flatten()(MaxPooling2D_5)
        Dense_1 = Dense(512, activation='relu')(Flatten_1)
        # Dropout_1 = Dropout(0.2)(Dense_1)
        # Dense_2 = Dense(512, activation='relu')(Dense_1)
        # Dropout_2 = Dropout(0.2)(Dense_2)
        # Dense_3 = Dense(1, activation='linear')(Dense_2)
    
        model = Model([Input_1], [Dense_1])
        return model
    
    def build_actor(self):
        state_input = Input(shape=(64,64,3,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))
        enc = self.get_cnn()
        x = enc(state_input)
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_actor_continuous(self):
        state_input = Input(shape=(64,64,3,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        enc = self.get_cnn()
        x = enc(state_input)
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, name='output', activation='tanh')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss_continuous(
                          advantage=advantage,
                          old_prediction=old_prediction)],experimental_run_tf_function=False)
        model.summary()

        return model

    def build_critic(self):

        state_input = Input(shape=(64,64,3,))
        enc = self.get_cnn()
        x = enc(state_input)
        x = Dense(HIDDEN_SIZE, activation='tanh')(x)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_value = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[out_value])
        model.compile(optimizer=Adam(lr=LR), loss='mse')

        return model

    def reset_env(self):
        self.episode += 1
        if self.episode % 50 == 0:
            self.val = True
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, 64,64,3), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:

            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_action_continuous(self):
        p = self.actor.predict([self.observation.reshape(1, 64,64,3), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)
        else:
            action = action_matrix = p[0]
        return action, action_matrix, p

    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            if CONTINUOUS is False:
                action, action_matrix, predicted_action = self.get_action()
            else:
                action, action_matrix, predicted_action = self.get_action_continuous()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

            if self.episode % 20 == 0:
                policy_model = self.actor
                policy_model.save('keras_models/model.h5')
            
            self.gradient_steps += 1


if __name__ == '__main__':
    ag = Agent()
    ag.run()