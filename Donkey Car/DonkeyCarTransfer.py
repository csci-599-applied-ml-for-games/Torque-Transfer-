import os
import sys
import random
import argparse
import signal

import numpy as np
import gym
import cv2

from matplotlib import pyplot as plt
from collections import deque
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

import gym_donkeycar

EPISODES = 10000
IMG_ROWS, IMG_COLS = 64, 64
IMG_CHANNELS = 3  # Based on black or white or color
SIM_PATH = './DonkeySimLinux/donkey_sim.x86_64'
MODEL_PATH = './saved_models/groudUpWeights.h5'

# Options
# 1 - Transfer learning model (pretrained model has batch norm)
# 2 - Transfer learning model (pretrained model has no batchnorm)
# 3 - Transfer learning model with canny (pretrained model has batchnorm)
# 4 - Non - Transfer Learning model
# 5 - Original Donkey car model (from repository)
MODEL_TYPE = 4


class DQNAgent:

    def __init__(self, state_size, action_space, train=True):
        self.t = 0
        self.max_Q = 0
        self.train = train

        # Get size of state and action
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = action_space

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if self.train:
            self.epsilon = 0.5
            self.initial_epsilon = 0.5
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        # Create replay memory using deque
        self.memory = deque(maxlen=10000)

        # Create main model and target model
        if MODEL_TYPE == 1:
            self.model = self.build_transfer_model_has_batchnorm()
            self.target_model = self.build_transfer_model_has_batchnorm()
        elif MODEL_TYPE == 2:
            self.model = self.build_transfer_model_has_no_batchnorm()
            self.target_model = self.build_transfer_model_has_no_batchnorm()
        elif MODEL_TYPE == 3:
            self.model = self.build_transfer_model_has_batchnorm_with_canny()
            self.target_model = self.build_transfer_model_has_batchnorm_with_canny()
        elif MODEL_TYPE == 4:
            self.model = self.build_dc_model_without_transfer()
            self.target_model = self.build_dc_model_without_transfer()
        else:
            self.model = self.build_og_dc_model_from_repo()
            self.target_model = self.build_og_dc_model_from_repo()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()
        self.tensorboard = TensorBoard(
            log_dir='./logs/',
            histogram_freq=0,
            write_graph=True,
            write_grads=True
        )
        self.tensorboard.set_model(self.model)

    def build_transfer_model_has_batchnorm(self):
        source_model = load_model('./files/my_model_new.h5')
        model = Sequential()
        for layer in source_model.layers[:-1]:
            if 'batch' not in layer.name:
                model.add(layer)
        for layer in model.layers:
            layer.trainable = True
        # model.add(Dense(512, activation="relu", name='dense'))
        # model.add(Dense(512, activation="relu", name='dense_1'))
        model.add(Dense(15, activation="linear", name='dense_2'))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def build_transfer_model_has_no_batchnorm(self):
        source_model = load_model('./files/my_model_gs.h5')
        model = Sequential()
        for layer in source_model.layers[:-3]:
            model.add(layer)
        for layer in model.layers:
            layer.trainable = True
        model.add(Dense(512, activation="relu", name='dense'))
        model.add(Dense(512, activation="relu", name='dense_1'))
        model.add(Dense(15, activation="linear", name='dense_2'))
        # print(model.summary())
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model
    
    def build_transfer_model_has_batchnorm_with_canny(self):
    #     source_model = load_model('./files/my_model_canny.h5')
    #     model = Sequential()
    #     for layer in source_model.layers[:-3]:
    #         model.add(layer)
    #     for layer in model.layers:
    #         layer.trainable = True
    #     model.add(Dense(512, activation="relu", name='dense'))
    #     model.add(Dense(512, activation="relu", name='dense_1'))
    #     model.add(Dense(15, activation="linear", name='dense_2'))
    #     # print(model.summary())
    #     adam = Adam(lr=self.learning_rate)
    #     model.compile(loss='mse', optimizer=adam)
    #     return model
    #
    # def build_og_dc_model_from_repo(self):
    #     model = Sequential()
    #     model.add(Conv2D(24, (5, 5), strides=(2, 2), padding="same",
    #                      input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))  # 80*80*4
    #     model.add(Activation('relu'))
    #     model.add(Conv2D(32, (5, 5), strides=(2, 2), padding="same"))
    #     model.add(Activation('relu'))
    #     model.add(Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    #     model.add(Activation('relu'))
    #     model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    #     model.add(Activation('relu'))
    #     model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    #     model.add(Activation('relu'))
    #     model.add(Flatten())
    #     model.add(Dense(512))
    #     model.add(Activation('relu'))
    #
    #     # 15 categorical bins for Steering angles
    #     model.add(Dense(15, activation="linear"))
    #
    #     adam = Adam(lr=self.learning_rate)
    #     model.compile(loss='mse', optimizer=adam)
    #
    #     return model
        return

    def build_dc_model_without_transfer(self):
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
        Dense_2 = Dense(512, activation='relu')(Dense_1)
        # Dropout_2 = Dropout(0.2)(Dense_2)
        Dense_3 = Dense(15, activation='linear')(Dense_2)

        model = Model([Input_1], [Dense_3])

        model.compile(optimizer='adam',
                      loss='mse')
        return model

    def rgb2gray(self, rgb):
        '''
        take a numpy rgb image return a new single channel image converted to greyscale
        '''
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def process_image(self, obs):
        # obs = self.rgb2gray(obs)
        obs = cv2.resize(obs, (IMG_ROWS, IMG_COLS))
        return obs

    def process_image_for_canny(self, obs):
        # obs = self.rgb2gray(obs)
        obs = cv2.resize(obs, (IMG_ROWS, IMG_COLS))
        obs1 = cv2.Canny(obs, 100, 200)
        return obs1

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()[0]
        else:
            print('Max Q')
            # print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)

            # Convert q array to steering value
            return linear_unbin(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon -
                             self.epsilon_min) / self.explore

    def named_logs(self, model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

    def train_replay(self, ep_num):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + \
                                          self.discount_factor * (target_val_[i][a])

        logs = self.model.train_on_batch(state_t, targets)
        return logs
        # self.tensorboard.on_epoch_end(ep_num, self.named_logs(self.model, [logs]))

    def update_tensorboard(self, e, log, rew):
        self.tensorboard.on_epoch_end(e, self.named_logs(self.model, [rew]))

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training

    def save_model(self, name):
        self.model.save_weights(name)


## Utils Functions ##

def linear_bin(a):
    """
    Convert a value to a categorical array.
    Parameters
    ----------
    a : int or float
        A value between -1 and 1
    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.
    See Also
    --------
    linear_bin
    """
    if not len(arr) == 15:
        raise ValueError('Illegal array length, must be 15')
    b = np.argmax(arr)
    a = b * (2 / 14) - 1
    return a


def run_ddqn(args):
    '''
    run a DDQN training session, or test it's result, with the donkey simulator
    '''

    # only needed if TF==1.13.1
    episode_wise_reward = []
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # Construct gym environment. Starts the simulator if path is given.
    env = gym.make(args.env_name, exe_path=SIM_PATH, port=args.port)

    # not working on windows...
    def signal_handler(signal, frame):
        print("catching ctrl+c")
        print(episode_wise_reward)
        plt.figure()
        plt.plot(episode_wise_reward)
        plt.show()
        env.unwrapped.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)

    # Get size of state and action from environment
    state_size = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
    action_space = env.action_space  # Steering and Throttle

    try:
        agent = DQNAgent(state_size, action_space, train=not args.test)

        throttle = args.throttle  # Set throttle as constant value

        episodes = []

        if os.path.exists(MODEL_PATH):
            print("load the saved model")
            agent.load_model(MODEL_PATH)
        log = 0
        for e in range(EPISODES):

            print("Episode: ", e)
            epi_reward = 0
            done = False
            obs = env.reset()

            episode_len = 0

            s_t = agent.process_image(obs)

            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*64*64*4
            while not done:

                # Get action for the current state and go one step in environment
                steering = agent.get_action(s_t)
                action = [steering, throttle]
                next_obs, reward, done, info = env.step(action)

                s_t1 = agent.process_image(next_obs)

                s_t1 = s_t1.reshape(1, s_t1.shape[0], s_t1.shape[1], s_t1.shape[2])  # 1x64x64x3

                # Save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(s_t, np.argmax(
                    linear_bin(steering)), reward, s_t1, done)
                agent.update_epsilon()

                if agent.train:
                    log = agent.train_replay(e)


                s_t = s_t1
                agent.t = agent.t + 1
                episode_len = episode_len + 1
                epi_reward+=reward
                if agent.t % 30 == 0:
                    print("EPISODE", e, "TIMESTEP", agent.t, "/ ACTION", action, "/ REWARD",
                          reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX ", agent.max_Q)

                if done:

                    # Every episode update the target model to be same with model
                    agent.update_target_model()

                    episodes.append(e)

                    # Save model for each episode
                    if agent.train:
                        agent.save_model(MODEL_PATH)

                    print("episode:", e, "  memory length:", len(agent.memory),
                          "  epsilon:", agent.epsilon, " episode length:", episode_len)
            episode_wise_reward.append(epi_reward)
            agent.update_tensorboard(e, log, epi_reward)




    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        env.unwrapped.close()


if __name__ == "__main__":
    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0"
    ]

    parser = argparse.ArgumentParser(description='ddqn')
    # parser.add_argument('--model', type=str,
    #                     default="rl_driver.h5", help='path to model')
    parser.add_argument('--test', action="store_true",
                        help='agent uses learned model to navigate env')
    parser.add_argument('--port', type=int, default=9091,
                        help='port to use for websockets')
    parser.add_argument('--throttle', type=float, default=0.3,
                        help='constant throttle for driving')
    parser.add_argument('--env_name', type=str, default='donkey-generated-track-v0',
                        help='name of donkey sim environment', choices=env_list)

    args = parser.parse_args()

    run_ddqn(args)
