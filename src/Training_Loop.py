"""
Author: Dudley Spence
Title: Solving the N-puzzle using Deep Reinforcement Learning

Training loop

This software trains the DQN agent through exploration and exploitation. This is the central script that
brings together the other elements, following the Deep Q-learning algorithm seen in the train_model(...) function.
"""
from Environment import Environment
from DQNAgent import DQNAgent
from Replay_Buffer import ReplayBuffer
from tqdm import tqdm
import tensorflow as tf
import math
import os
from collections import deque
#os.environ["TF_USE_LEGACY_KERAS"] = "1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found, using CPU.")




def collect_gameplay_experiences(env, agent, buffer, state_history):
    """
    Collects gameplay experiences by playing one step of a puzzle
    and stores the gameplay experiences in buffer.
    :param env: The gameplay environment
    :param agent: The DQN agent
    :param buffer: The replay buffer
    :return: done: Boolean
    """
    state_hot = agent.convert_board_one_hot(env.current_state)
    state = tf.reshape(state_hot, (1, -1))
    action = agent.epsilon_greedy_policy(state)
    next_state, reward, done = env.apply_action(action)
    next_state_hot = agent.convert_board_one_hot(next_state)

        # Loop detection and penalization
    state_hash = hash(next_state_hot.tostring())
    if state_hash in state_history:
        reward -= 10  # Penalize the loop
    state_history.append(state_hash)
    
    buffer.store_gameplay_experience(state_hot, next_state, reward, action, done)
    return done


def train_model(max_episodes=10000, n=3, batch_size=100, start_difficulty=5, final_epsilon=0.3, initial_epsilon=0.9,
                nodes=250,
                gamma=0.9, buffer_size=100000, learning_rate=0.0001, time_steps=50):
    """
    This function follows the DQN training algorithm and is the main training loop.
    :param max_episodes: The maximum number of training epochs
    :param n: the number of tiles per row and column
    :param batch_size: the size of the sample batches
    :param difficulty: the desired estimate for number of actions to optimally solve the puzzles
    :param final_epsilon: the final epsilon value once epsilon decay has finished
    :param initial_epsilon: the initial epsilon for a new untrained DQN
    :param nodes: the number of nodes per layer in the network
    :param gamma: The discount factor
    :param buffer_size: the size of the replay buffer
    :param learning_rate: The learning rate
    :param time_steps: The number of time-steps in each epoch
    :return: None
    """
    epsilon_decay_rate = 0.999
    current_difficulty = start_difficulty    
    with tf.device('/GPU:0'):
        agent = DQNAgent(final_epsilon, initial_epsilon, n, start_difficulty, nodes, gamma, learning_rate, summary=True)
        buffer = ReplayBuffer(buffer_size)
        env = Environment(n)
        total_finishes = 0


        # Use tf.data.Dataset for efficient data handling
        dataset = tf.data.Dataset.from_generator(lambda: buffer.sample_gameplay_batch(batch_size),
                                                 output_types=(tf.float32, tf.float32, tf.float32, tf.int32, tf.bool))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


        state_history = deque(maxlen=10)
        
        for episode_cnt in tqdm(range(max_episodes)):
            agent.epsilon = agent.epsilon if agent.epsilon < final_epsilon else agent.epsilon * epsilon_decay_rate
            env.generate_new_board(current_difficulty)

            
            
            for i in range(time_steps):
                done = collect_gameplay_experiences(env, agent, buffer, state_history)
                if episode_cnt >= 100:
                    gameplay_experience_batch = buffer.sample_gameplay_batch(batch_size)
                    loss = agent.train(gameplay_experience_batch)
                if done:
                    break
                if time_steps % 10 == 0:
                    agent.update_target_network()
            if episode_cnt >= 100 and episode_cnt % 50 == 0:
                done_count, just_completed = evaluate_training_result(env, agent, current_difficulty)
                total_finishes += done_count
                print("total finishes is {0}".format(total_finishes))
                print("so far the loss is {0}".format(loss))
                print('\n')
                if ((current_difficulty < 22) & (just_completed == 100)):
                    current_difficulty += 1
                    agent.difficulty = current_difficulty



def evaluate_training_result(env, agent, current_difficulty):
    """
    Evaluates the in-situ performance of the current DQN agent by using it to play
    10 puzzles with 50 actions per game to solve the puzzle.
    The higher the average reward is the better the DQN agent performs.
    :param difficulty: the estimated number of actions to optimally solve the puzzle
    :param env: the game environment
    :param agent: the DQN agent
    :return: done_count: the number of puzzles solved out of the 10
    """
    done_count = 0.0
    final_distance = 0
    episodes_to_play = 10
    for i in range(episodes_to_play):
        env.generate_new_board(current_difficulty)
        for j in range(50):
            state = agent.convert_board_one_hot(env.current_state)
            state = tf.reshape(state, (1, -1))
            action = agent.policy(state)
            next_state, reward, done = env.apply_action(action)
            if done:
                done_count += 1
                break
        final_distance += env.manhattan_linear_conflict()
    avg_final_distance = final_distance / episodes_to_play

    just_completed = done_count * 10
    print('\n')
    print(f"{agent.N - 1}-Puzzle")
    print("Difficulty: {0}".format(current_difficulty))
    print("epsilon is {0}".format(agent.epsilon))
    print(f'Just solved {just_completed}%')
    print("so far the avg final distance is {0}".format(avg_final_distance))

    return done_count, just_completed


# Below is the user input part of the script
# Basic parameters for training are taken from the user

N = int(input("What is the value of N for the puzzle you would like to train? "))
n = int(math.sqrt(N + 1))
if n > 3:
    nodes = 500
    final_epsilon = 0.3
else:
    nodes = 250
    final_epsilon = 0.4
difficulty = 5
# difficulty = int(input("For what difficulty would you like to start training the puzzle from? "))
restart = "hi"
while restart[0] not in ["Y", "y", "N", "n"]:
    restart = input("Would you like to begin training from scratch? Y/N ")

if restart.upper() == "Y":
    if os.path.exists("training/training_" + str(N) + "_puzzle_" + str(nodes) + "_nodes"):
        os.system("rm -rf " + "training/training_" + str(N) + "_puzzle_" + str(nodes) + "_nodes")

else:
    if not os.path.exists("training/training_" + str(N) + "_puzzle_" + str(nodes) + "_nodes"):
        print("There is no record of previous training therefore training will restart.")


# To further optimize the hyper-parameters change the function parameters below
train_model(max_episodes=100000, n=n, batch_size=500, start_difficulty=5,
            final_epsilon=0.3, initial_epsilon=0.9, nodes=nodes, gamma=0.9, buffer_size=500000,
            learning_rate=0.0001, time_steps=50)
