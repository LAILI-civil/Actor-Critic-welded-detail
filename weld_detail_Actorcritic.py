import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import numpy as np
import matplotlib.pyplot as plt

def corrosion(year):
    a = 92.6
    b = 0.75
    thickness = (a * year ** b) / 1e3
    return thickness


class environment():
    """this part define the bridge degradation process"""
    def __init__(self):
        """Component state transition"""
        self.component_tran1 = np.array([[[0.964, 0.036, 0, 0, 0],
                                          [0, 0.9492, 0.0508, 0, 0],
                                          [0, 0, 0.974, 0.026, 0],
                                          [0, 0, 0, 0.978, 0.022],
                                          [0, 0, 0, 0, 1]
                                          ]])
        self.component_tran2 = np.array([[[0.937, 0.063, 0, 0, 0],
                                          [0, 0.912, 0.088, 0, 0],
                                          [0, 0, 0.954, 0.046, 0],
                                          [0, 0, 0, 0.961, 0.039],
                                          [0, 0, 0, 0, 1]
                                          ]])
        self.component_tran3 = np.array([[[0.9115, 0.0885, 0, 0, 0],
                                          [0, 0.8778, 0.1222, 0, 0],
                                          [0, 0, 0.9351, 0.0649, 0],
                                          [0, 0, 0, 0.9447, 0.0553],
                                          [0, 0, 0, 0, 1]
                                          ]])
        self.component_tran4 = np.array([[[0.8873, 0.1127, 0, 0, 0],
                                          [0, 0.846, 0.154, 0, 0],
                                          [0, 0, 0.9168, 0.0832, 0],
                                          [0, 0, 0, 0.929, 0.071],
                                          [0, 0, 0, 0, 1]
                                          ]])

        self.component_transition = np.concatenate(
            (self.component_tran1, self.component_tran2, self.component_tran3, self.component_tran4), axis=0)
        # Rehabilitation action
        self.Rehabilitation = np.array([[0.99, 0.01, 0, 0, 0],
                                        [0, 0.99, 0.01, 0, 0],
                                        [0, 0, 0.99, 0.01, 0],
                                        [0, 0, 0, 0.99, 0.01],
                                        [0, 0, 0, 0, 1]
                                        ])

        # repair action
        self.repair = np.array([[0.9, 0.1, 0, 0, 0],
                                [0.9, 0.1, 0, 0, 0],
                                [0.9, 0.1, 0, 0, 0],
                                [0.9, 0.1, 0, 0, 0],
                                [0.9, 0.1, 0, 0, 0]
                                ])

        # Observed with SHM data
        N = np.array([0, 3523610, 5920256, 10054128, 14829615])
        Obeservation_matrix = np.zeros([5, 5])
        prob = np.zeros(4)
        for i in range(len(N)-1):
            prob[i] = N[i+1] - N[i] + 1

        for step, ri in enumerate(prob):
            Pi = ri / (1 + ri)
            Qi = 1 / (1 + ri)
            Obeservation_matrix[step, step] = Pi
            Obeservation_matrix[step, step + 1] = Qi
        Obeservation_matrix[4, 4] = 1

        self.Obeservation_matrix = Obeservation_matrix

        # define the baseline for cycling
        self.mean_Cycling_num = 75823.19
        self.variance_Cycling_num = 383.4823 * 3

        # define the year for corrosion
        self.paint = 3
        self.year = 0

        # Reward table is the immediate reward based on the states (column) and actions (row)
        self.Reward_table = np.array([[0, -2, -4, -8, -400],
                                      [-2, -4, -6, -10, -400],
                                      [-10, -12, -14, -18, -400],
                                      [-200, -200, -200, -200, -400],
                                      ])
        self.state = []

    def reset(self):
        self.state = np.array([
            [0.9, 0.1, 0, 0, 0],
        ])


    def step(self,states,action,hidden_state,paint,year):
        """"""
        if action == 0:
            paint = paint - 1
            if paint < 0:
                paint = 0
                year = year + 1
            mean_Adjuest_num = self.mean_Cycling_num / (1 - 2 * corrosion(year) / 10) ** 5
            variance_Adjuest_num = self.variance_Cycling_num * (1 / (1 - 2 * corrosion(year) / 10) ** 5) ** 2
            Cum_current = np.random.normal(mean_Adjuest_num, variance_Adjuest_num, 1)
            observation_T = np.linalg.matrix_power(self.Obeservation_matrix, np.int(Cum_current))
            if year < 15:
                transition_T = self.component_transition[0, :, :]
            elif year >= 15 and year < 30:
                transition_T = self.component_transition[1, :, :]
            elif year >= 30 and year < 45:
                transition_T = self.component_transition[2, :, :]
            else:
                transition_T = self.component_transition[3, :, :]

        elif action == 1:
            paint = self.paint
            paint = paint - 1
            mean_Adjuest_num = self.mean_Cycling_num / (1 - 2 * corrosion(year) / 10) ** 5
            variance_Adjuest_num = self.variance_Cycling_num * (1 / (1 - 2 * corrosion(year) / 10) ** 5) ** 2
            Cum_current = np.random.normal(mean_Adjuest_num, variance_Adjuest_num, 1)
            observation_T = np.linalg.matrix_power(self.Obeservation_matrix, np.int(Cum_current))
            if year < 15:
                transition_T = self.component_transition[0, :, :]
            elif year >= 15 and year < 30:
                transition_T = self.component_transition[1, :, :]
            elif year >= 30 and year < 45:
                transition_T = self.component_transition[2, :, :]
            else:
                transition_T = self.component_transition[3, :, :]

        elif action == 2:
            paint = self.paint
            paint = paint - 1
            observation_T = np.eye(5)
            if year < 15:
                transition_T = self.component_transition[0, :, :]
            elif year >= 15 and year < 30:
                transition_T = self.component_transition[1, :, :]
            elif year >= 30 and year < 45:
                transition_T = self.component_transition[2, :, :]
            else:
                transition_T = self.component_transition[3, :, :]

        elif action == 3:
            paint = self.paint
            year = 0
            paint = paint - 1
            observation_T = self.repair
            transition_T = self.repair

        Transition_posteriori = (transition_T + observation_T) / 2

        Random_number = np.random.uniform(0, 1)
        new_hidden_state = np.zeros((1, 1), dtype=int)
        state_mark = 0.
        for i in range(5):
            state_mark = state_mark + Transition_posteriori[hidden_state, i]
            if Random_number <= state_mark:
                new_hidden_state = i
                break


        # Update belief state
        new_state = states @ Transition_posteriori

        #immediate reward part
        reward = 0
        if action == 0:
            for i in range(5):
                reward = reward + states[0, i] * self.Reward_table[0, i]

        elif action == 1:
            for i in range(5):
                reward = reward + states[0, i]*self.Reward_table[1, i]

        elif action == 2:
            for i in range(5):
                reward = reward + states[0, i] * self.Reward_table[2, i]

        elif action == 3:
            for i in range(5):
                reward = reward + states[0, i] * self.Reward_table[3, i]
        else:
            print('error')

        return new_state, reward, new_hidden_state, paint, year

def Actor_build(n_actions,learning_rate=1e-4,state_shape = [None,5]):
    Actor_network = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(n_actions),
    ])

    Actor_network.build(state_shape)
    Actor_network.optimizer = optimizers.Adam(learning_rate)
    return Actor_network


def Actor_learn(Actor_network,state,action,TD_error):
    with tf.GradientTape() as tape:
        logit = Actor_network(state)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[action], logits=logit)
        loss_actor = tf.reduce_sum(tf.multiply(cross_entropy, TD_error))
    grads = tape.gradient(loss_actor, Actor_network.trainable_variables)
    Actor_network.optimizer.apply_gradients(zip(grads, Actor_network.trainable_variables))
    return Actor_network


def get_action(Actor_network,state,greedy=False):
    logit = Actor_network(state)
    prob = tf.nn.softmax(logit).numpy()
    if greedy:
        return np.argmax(prob.ravel())
    action = np.random.choice(logit.shape[1], p=prob.ravel())
    return action


def Critic_build(learning_rate=1e-3,state_shape = [None,5]):
    Critic_network = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(1),
    ])
    Critic_network.build(state_shape)
    Critic_network.optimizer = optimizers.Adam(learning_rate)
    return Critic_network


def Critic_learn(Critic_network,state_old,reward,state_new,gamma):
    with tf.GradientTape() as tape:
        v_old = Critic_network(state_old)
        v_new = Critic_network(state_new)
        TD_error = reward + gamma * v_new - v_old
        loss_critic = tf.square(TD_error)
    grads = tape.gradient(loss_critic, Critic_network.trainable_variables)
    Critic_network.optimizer.apply_gradients(zip(grads, Critic_network.trainable_variables))
    return TD_error,Critic_network


class Agent():
    def __init__(self,
                 Actor,
                 Critic,
                 n_actions=8,
                 input_shape=15,
                 batch_size=32,
                 max_states=350000
                 ):
        # state vector and action number
        self.n_actions = n_actions
        self.input_shape = input_shape

        # define the Actor-Critic networks
        self.Actor = Actor
        self.Critic = Critic

    def save(self,folder_name, **kwargs):
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save Main_DQN and Target_DQN
        self.Actor.save(folder_name + '/dqn.h5')
        self.Critic.save(folder_name + '/target_dqn.h5')

def main():
    initial_states = np.array([[0.9, 0.1, 0, 0, 0]])
    Actor_network = Actor_build(4)
    Critic_network = Critic_build()
    Environment = environment()
    num_episode = 3000
    max_over_step = 200
    gamma = 0.95
    agent = Agent(Actor_network, Critic_network)
    initial_hidden_state = 0
    t_plot = []
    reward_plot = []
    for i in range(0, num_episode):
        t = 0
        states = initial_states
        hidden_state = initial_hidden_state
        reward_sum = 0
        paint = 3
        year = 0
        action_episode_1 = np.zeros((201,))
        hidden_state_episode_1 = np.zeros((201,))

        while t <= max_over_step:

            action = get_action(Actor_network, states)

            action_episode_1[t] = action
            hidden_state_episode_1[t] = hidden_state

            New_belief_state, reward, hidden_state, paint, year = Environment.step(states, action, hidden_state, paint, year)

            TD_error, Critic_network = Critic_learn(Critic_network, states, reward, New_belief_state, gamma)
            Actor_network = Actor_learn(Actor_network, states, action, TD_error)

            states = New_belief_state
            reward_sum = reward_sum + reward

            t += 1
            if t == max_over_step:
                print("epoch num:", i, " time step: ", t, "   TD_error: ", float(TD_error.numpy()), "   Reward: ", reward_sum)
                print(action, np.around(states, 3), hidden_state)
                print("------------------------------------------------------------------------------------------")
                plt.ion()
                fig1 = plt.figure(1, figsize=(7, 4))
                fig1.canvas.manager.window.move(200, 200)
                plt.clf()
                plt.plot(np.linspace(0, 200, 201), action_episode_1, c="green", alpha=0.5, label='Action',linewidth=0.6)
                plt.plot(np.linspace(0, 200, 201), hidden_state_episode_1, label='Hidden state',
                         color='deepskyblue')
                plt.legend(loc='best', fontsize=8)
                plt.xlabel("life-cycle(year)", fontsize=8)
                plt.ylabel("hidden state", fontsize=8)
                plt.tight_layout()
                plt.draw()
                fig1.savefig('D:/figure/temp{}.png'.format(i), dpi=300)
                plt.pause(0.1)

                fig2 = plt.figure(2, figsize=(7, 4))
                fig2.canvas.manager.window.move(1000, 200)
                plt.clf()
                plt.ylim(-10000, 0)
                t_plot.append(i)
                reward_plot.append(reward_sum)
                plt.plot(t_plot, reward_plot, label='Sum reward', color='blueviolet', alpha=1, linewidth=0.4)
                plt.legend(loc='best', fontsize=8)
                plt.xlabel("Episode", fontsize=8)
                plt.ylabel("Sum reward", fontsize=8)
                plt.show()

if __name__ == '__main__':
    main()




