import numpy as np
import argparse
import torch
from copy import deepcopy
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import signal
import cProfile
import pickle

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn
import matplotlib.pyplot as plt
plt.set_loglevel("error")

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor
from logger import Logger

#sys.path.append("/home/brendon/Research/FAIRIS/")
#os.chdir("/home/brendon/Research/FAIRIS/")
#sys.path.append("/FAIRIS/")
#os.chdir("/FAIRIS/")
sys.path.append("/home/b/brendon45/FAIRIS/")
os.chdir("/home/b/brendon45/FAIRIS/")

from FAIRISEnvTF import FAIRISEnvTF

import time

# Signal debugging stuff
#sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
#for sig in sigs:
#    signal.signal(sig, print)

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
#parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
parser.add_argument('--learning-rate',type=float, default=.0005, help='Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=50000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=4, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(250000), help='number of maximum steps to take.')# 250000 # bout 4 million
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='/home/b/brendon45/oc_tests/runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')
parser.add_argument('--maze', type=str, default='fourrooms')
parser.add_argument('--state_size', type=int, default=1024)
parser.add_argument('--sequence', type=bool, default=False)
parser.add_argument('--horizon', type=int, default=1000)
parser.add_argument('--name', type=str)
parser.add_argument('--reward_len', type=int, default=1)
parser.add_argument('--pc_file', type=str, default='uniform_32_l')

def draw_path_v2(path, options, filename, walls, goal, subgoals, starting):
    plt.clf()
    fig, ax = plt.subplots()

    for wall in walls:
        ax.plot(wall[0], wall[1], 'black')

    #ax.scatter(starting[0], starting[1], c='r')
    #if subgoals is None:
    #    ax.scatter(goal[0], goal[1], c='g')
    #else:
    #    sub = np.asarray(subgoals)
    #    ax.scatter(sub[:, 0], sub[:, 1], c='g')

#    print(f"path: {path}")
    data = np.asarray(path)
    ax.plot(data[:, 0], data[:, 1], c='black')

    ops = {'0': [], '1': [], '2': [], '3': []}
    ops_c = {'0': 'b', '1': 'y', '2': 'orange', '3': 'cyan'}
    for idx in range(len(path)):
        ops[str(options[idx])].append(path[idx])

#    print(f"ops: {ops}")
    for key in ops.keys():
        if len(ops[key]) > 0:
            o_data = np.asarray(ops[key])
            ax.scatter(o_data[:, 0], o_data[:, 1], c=ops_c[key])

    ax.relim()
    ax.autoscale_view()
    plt.savefig(filename)
    plt.close()

def draw_path(path, options, filename, walls, borders):
    plt.clf()
    fig, ax = plt.subplots()
    rect = patches.Rectangle(borders[0], borders[1][0], borders[1][1], linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.relim()
    ax.autoscale_view()

    for wall in walls:
        ax.plot(wall[0], wall[1], 'black')

    # Plot path
    data = np.asarray(path)
    ax.plot(data[:, 0], data[:, 1], 'g')

    o_0 = []
    o_1 = []
    o_2 = []
    o_3 = []
    for idx in range(len(path)):
        if options[idx] == 0:
            o_0.append(path[idx])
        elif options[idx] == 1:
            o_1.append(path[idx])
        elif options[idx] == 2:
            o_2.append(path[idx])
        else:
            o_3.append(path[idx])

    d0 = np.asarray(o_0)
    d1 = np.asarray(o_1)
    d2 = np.asarray(o_2)
    d3 = np.asarray(o_3)

    if not(len(o_0) <= 0):
        ax.scatter(d0[:, 0], d0[:, 1], c='g')
    if not(len(o_1) <= 0):
        ax.scatter(d1[:, 0], d1[:, 1], c='y')
    if not(len(o_2) <= 0):
        ax.scatter(d2[:, 0], d2[:, 1], c='r')
    if not(len(o_3) <= 0):
        ax.scatter(d3[:, 0], d3[:, 1], c='b')

    # Save file
    plt.savefig(filename)

def run(args):
    # Parameters
#    maze_file = "/home/b/brendon45/oc_tests/tworooms.xml"
#    maze_file = "/home/b/brendon45/oc_tests/fourrooms.xml"
#    file_name = "/home/b/brendon45/oc_tests/uniform_16"
    #maze_name = "fourrooms_obs"
    maze_name = args.maze
    maze_file = f"/home/b/brendon45/oc_tests/mazes/{maze_name}.xml"
    file_name = f"/home/b/brendon45/oc_tests/pc_grids/{args.pc_file}"
    state_size = args.state_size#1024#256#1024
    num_actions = 8
    horizon = args.horizon#500#1000
    test_int = 5
    test_lens = []
    last_test = -1
    terms = []
    greedys = []
    lens_list = []
    steps_list = []
    sequence_learning = args.sequence
    walls = [[[-2,2], [0,0]], [[0,0], [-2,2]], [[-5,-3], [0,0]], [[5,3], [0,0]], [[0,0], [-5,-3]], [[0,0], [5,3]]]
    border = [(-5, -5), (10, 10)]
    lowest_run = horizon + 10

    env = FAIRISEnvTF(maze_file, horizon, file_name, sequence_learning, args.reward_len)
#    print(f"Env subgoals: {env.robot.maze.subgoals}, goals: {env.robot.maze.goal_location}")
    option_critic = OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f"Device is: {device}")
    print(f"Maze file is: {maze_file}")
    print(f"Args are: {args}")

    option_critic = option_critic(
        in_features=state_size,
        num_actions=num_actions,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
#    env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.exp}-{time.ctime()}")

    steps = 0 ;
    test_step = 0
    if args.switch_goal: print(f"Current goal {env.goal}")
    while steps < args.max_steps_total:


        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}

        env.start_path()

        obs   = env.reset()
        state = option_critic.get_state(to_tensor(obs))
        greedy_option  = option_critic.greedy_option(state)
        current_option = 0
        losses = []

        # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
        # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
        # should be finedtuned (this is what we would hope).

        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        while not done and ep_steps < args.max_steps_ep:
            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0
    
            action, logp, entropy = option_critic.get_action(state, current_option)
            env.set_option(current_option)

            next_obs, reward, done = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic_prime, args.gamma, args.termination_reg, args.entropy_reg)
                loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args.gamma)
                    loss += critic_loss

                losses.append(np.absolute(loss.detach().cpu().numpy().item()))
                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

        path, ops = env.get_path()
        if len(path) < lowest_run:
            path_name = f"/home/b/brendon45/oc_tests/images/{maze_name}_{args.name}_s.png" 
            walls, goal, subgoals, starting = env.robot.maze.get_plot_data()
            draw_path_v2(path, ops, path_name, walls, goal, subgoals, starting)
            lowest_run = len(path)

        lens_list.append(ep_steps)
        steps_list.append(steps)

        logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon, env.robot.maze.current_goal, np.mean(losses))

#        if (test_step % test_int) == 0:
            # Run evalulation
#            env.start_path()
#            obs   = env.reset()
#            state = option_critic.get_state(to_tensor(obs))
#            greedy_option  = option_critic.greedy_option(state)
#            current_option = 0
#            done = False
#            option_termination = True
#            length = 0
#            terms = []
#            greedys = []
#            while not done:
#                if option_termination:
#                    current_option = greedy_option
#                env.set_option(current_option)

#                action, logp, entropy = option_critic.get_action(state, current_option)
#
#                next_obs, reward, done = env.step(action)
#
#                state = option_critic.get_state(to_tensor(next_obs))
#                option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)
#                terms.append(option_termination)
#                greedys.append(greedy_option)
#                obs = next_obs
#                length += 1
#
#            test_lens.append(length)
#            print(f"Test length: {length}")
#            path, ops = env.get_path()
#            if length < lowest_run:
#                lowest_run = length
#                path_name = f"/home/b/brendon45/oc_tests/images/{maze_name}_{args.name}_s.png" 
#                draw_path(path, ops, path_name, walls, border)
#            else:
#                path_name = f"/home/b/brendon45/oc_tests/images/{maze_name}_{args.name}_c.png" 
#                draw_path(path, ops, path_name, walls, border)
#                
##        env.robot.plot_goal_lengths()
#
#        test_step += 1

    path_lengths = {"lengths": lens_list, "steps": steps_list}
    with open(f"/home/b/brendon45/oc_tests/data/path_{args.name}_lengths.pkl", 'wb') as fp:
        pickle.dump(path_lengths, fp)

    # Save path
#    print(f"Path: {path}, ops: {ops}, terms: {terms}, greedys: {greedys}")
    walls = [[[-2,2], [0,0]], [[0,0], [-2,2]], [[-5,-3], [0,0]], [[5,3], [0,0]], [[0,0], [-5,-3]], [[0,0], [5,3]]]
    border = [(-5, -5), (10, 10)]
#    border = [(-4, -2), (8, 4)]
#    walls = [[[-4, 3], [0, 0]]]
#    draw_path(path, ops, "/home/b/brendon45/oc_tests/path.png", walls, border)
    # Save graph
    plt.clf()
    plt.plot(test_lens)
    plt.savefig(f'/home/b/brendon45/oc_tests/images/test_lens_{args.name}.png')
    env.closeSim(0)

if __name__== "__main__":
    args = parser.parse_args()
    #cProfile.run('run(args)', filename='/home/b/brendon45/oc_tests/profile.txt')
    run(args)
