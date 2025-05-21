import numpy as np
import torch
from copy import deepcopy
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

from option_critic_no_term import OptionCriticFeatures, OptionCriticConv
from option_critic_no_term import critic_loss as critic_loss_fn
from option_critic_no_term import actor_loss as actor_loss_fn
import matplotlib.pyplot as plt
plt.set_loglevel("error")

from experience_replay import ReplayBuffer
from utils import make_env, to_tensor

sys.path.append("/home/b/brendon45/FAIRIS/")
os.chdir("/home/b/brendon45/FAIRIS/")

from FAIRISEnvTF import FAIRISEnvTF

import datetime
import time

def draw_path_v2(path, options, filename, walls, goal, subgoals, starting):
    plt.clf()
    fig, ax = plt.subplots()

    for wall in walls:
        ax.plot(wall[0], wall[1], 'black')

    data = np.asarray(path)
    ax.plot(data[:, 0], data[:, 1], c='black')

    ops = {'0': [], '1': [], '2': [], '3': []}
    ops_c = {'0': 'b', '1': 'y', '2': 'orange', '3': 'cyan'}
    for idx in range(len(path)):
        ops[str(options[idx])].append(path[idx])

    for key in ops.keys():
        if len(ops[key]) > 0:
            o_data = np.asarray(ops[key])
            ax.scatter(o_data[:, 0], o_data[:, 1], c=ops_c[key])

    ax.relim()
    ax.autoscale_view()
    plt.savefig(filename)
    plt.close()

def get_terminations(border, option_critic, env, num_ops):
    data_x = []
    data_y = []

    op_terms = {}
    greedy_ops = {}
    for idx in range(num_ops):
        op_terms[f"{idx} on_x"] = []
        op_terms[f"{idx} on_y"] = []
        op_terms[f"{idx} off_x"] = []
        op_terms[f"{idx} off_y"] = []
        greedy_ops[f"{idx}:x"] = []
        greedy_ops[f"{idx}:y"] = []

    w = border[0][0]
    h = border[0][1]
    points = 200
    num_rows = points / h
    num_cols = points / w
    inc_x = w / num_cols
    inc_y = h / num_rows

    cur_y = border[1][1]
    for idy in range(int(num_rows)):
        cur_x = border[1][0]
        for idx in range(int(num_cols)):
            cur_pcs = env.pc_net.get_all_pc_activations_normalized(cur_x, cur_y)
            cur_state = option_critic.get_state(to_tensor(cur_pcs))
            terminations = option_critic.get_terminations(cur_state).detach().cpu().numpy()[0]
            #print(f"Terminations are: {terminations}")
            greedy_op = option_critic.greedy_option(cur_state)

            for idx in range(num_ops):
                if terminations[idx] >= 0.5:
                    op_terms[f"{idx} on_x"].append(cur_x) 
                    op_terms[f"{idx} on_y"].append(cur_y) 
                else:
                    op_terms[f"{idx} off_x"].append(cur_x) 
                    op_terms[f"{idx} off_y"].append(cur_y) 
            greedy_ops[f"{greedy_op}:x"].append(cur_x)

            greedy_ops[f"{greedy_op}:y"].append(cur_y)

            cur_x += inc_x
        cur_y += inc_y

    return op_terms, greedy_ops
    

def run(args, env):
    # Parameters
    maze_name = args["maze_name"]
    maze_file = f"/home/b/brendon45/oc_tests/mazes/{maze_name}.xml"
    file_name = f"/home/b/brendon45/oc_tests/pc_grids/{args['pc_file']}"
    state_size = args['state_size']
    num_actions = 8
    lens_list = []
    steps_list = []
    lowest_run = horizon + 10
    shortest_path = []
    shortest_ops = []
    count = 0
    short_time = 0
    short_op_len_ratio = 0
    ave_op_length = []
    op_len_ratio = []
    options_lengths_ave = {}
    shortest_path_time = 0
    shortest_ol_ratio = None
    shortest_op_lengths = None
    max_lengths = []
    for op in range(args["num_options"]):
        options_lengths_ave[op] = []

    option_critic = OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    option_critic = option_critic(
        in_features=state_size,
        num_actions=num_actions,
        num_options=args["num_options"],
        term_val=args["term_val"],
        temperature=args["temp"],
        eps_start=args["epsilon_start"],
        eps_min=args["epsilon_min"],
        eps_decay=args["epsilon_decay"],
        eps_test=args["optimal_eps"],
        device=device
    )
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)

    optim = torch.optim.RMSprop(option_critic.parameters(), lr=args["learning_rate"])

    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
#    env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args["max_history"], seed=args["seed"])

    steps = 0 ;
    converged = False
    while steps < args["max_steps_total"] and not(converged):

        rewards = 0 ; option_lengths = {opt:[] for opt in range(args["num_options"])}

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
        op_lengths = []
        last_term = 0
        while not done:
            epsilon = option_critic.epsilon

            thres = np.exp(-0.0001*steps)
            rand_n = np.random.random()

#            if option_termination:
            if option_termination and (rand_n > thres or last_term > 2):
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args["num_options"]) if np.random.rand() < epsilon else greedy_option
                op_lengths.append(curr_op_len)
                curr_op_len = 0
                last_term = 0
    
            action, logp, entropy = option_critic.get_action(state, current_option)
            env.set_option(current_option)

            next_obs, reward, done = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args["batch_size"]:
                actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                    reward, done, next_obs, option_critic, option_critic_prime, args["gamma"], args["term_reg"], args["entropy_reg"])
                loss = actor_loss

                if steps % args["update_frequency"] == 0:
                    data_batch = buffer.sample(args["batch_size"])
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args["gamma"])
                    loss += critic_loss

                losses.append(np.absolute(loss.detach().cpu().numpy().item()))
                optim.zero_grad()
                loss.backward()
                optim.step()

                if steps % args["freeze_interval"] == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            state = option_critic.get_state(to_tensor(next_obs))
            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs
            last_term += 1

        ave_op_length.append(np.mean(op_lengths))
        if len(op_lengths) > 0:
            max_op_len = np.amax(op_lengths)
        else:
            max_op_len = ep_steps
        if max_op_len == 0:
            max_op_len = ep_steps
        max_lengths.append(max_op_len)

        # Calculate option length ratio
        option_len_ratio = (max_op_len * args["num_options"]) / ep_steps
        op_len_ratio.append(option_len_ratio)

        if count % 1 == 0:
            print(f"{datetime.datetime.now()} Current step: {steps}, cur_len: {ep_steps}, loss: {np.mean(losses):.3f}, eps: {option_critic.epsilon:.3f}, op_len_ratio: {option_len_ratio:.3f}, ave_op_len: {np.mean(op_lengths)}")

        path, ops = env.get_path()
        if len(path) < lowest_run:
            shortest_path = path
            shortest_ops = ops
            short_time = 0
            lowest_run = len(path)
            short_op_len_ratio = option_len_ratio
            #border = [(7, 3), (-3.5, -1.5)]
            border = [(10, 10), (-5, -5)]
            op_terms, greedy_ops = get_terminations(border, option_critic, env, args["num_options"])
            shortest_path_time = steps
            shortest_ol_ratio = option_len_ratio
            shortest_op_lengths = max_op_len
#            comb_dict = {"op_terms": op_terms, "greedy_ops": greedy_ops}
#            with open(f"/home/b/brendon45/oc_tests/data/plot_terms_test.pkl", 'wb') as fp:
#                pickle.dump(comb_dict, fp)
                

        for idx in range(args["num_options"]):
            if len(option_lengths[idx]) > 0:
                options_lengths_ave[idx].append(np.mean(option_lengths[idx]))
            else:
                options_lengths_ave[idx].append(None)

        lens_list.append(ep_steps)
        steps_list.append(steps)
        count += 1
        short_time += 1
        if (lowest_run < 50) and (short_time > 20):
            converged = True
            print(f"Lowest run is {lowest_run}")

        if len(path) > 50:
            short_time = 0

    convergence_time = steps

    return lens_list, steps_list, shortest_path, shortest_ops, shortest_path_time, shortest_ol_ratio, shortest_op_lengths, convergence_time, options_lengths_ave, ave_op_length, op_len_ratio, short_op_len_ratio, op_terms, greedy_ops, max_lengths

if __name__== "__main__":
    args = { # 250000
        "optimal_eps": 0.05, "learning_rate": 0.0005, "gamma": 0.99, 
        "epsilon_start": 1.0, "epsilon_min": 0.0, "epsilon_decay": 50000,
        "max_history": 10000, "batch_size": 32, "freeze_interval": 200, 
        "update_frequency": 4, "term_reg": 0.01, "entropy_reg": 0.01,
        "num_options": 4, "temp": 1, "max_steps_total": 350000, "seed": 0,
        "maze_name": "fourrooms", "state_size": 1024, "horizon": 1000, "pc_file": "uniform_32_l",
        "noise": False, "noise_std": 0.0, "noise_mean": 0.0, "term_val": 1.0,
    }
    print(f"Args: {args}")

    maze_file = f"/home/b/brendon45/oc_tests/mazes/{args['maze_name']}.xml"
    file_name = f"/home/b/brendon45/oc_tests/pc_grids/{args['pc_file']}"
    horizon = args["horizon"]
    env = FAIRISEnvTF(maze_file, horizon, file_name, False, 1, args['noise'], args['noise_std'], args['noise_mean'])
    experiment_name = f"{args['maze_name']}_no_term_experiment_100000"
    print(experiment_name)

#    test_values = [0.00, 0.01, 0.03, 0.05, 0.07, 0.09]
    test_values = [0.5, 0.5, 0.5, 0.5, 0.5]
#    test_values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0]

    seed = time.time_ns() % (2**32)
    print(f"Seed is: {seed}")
    args['seed'] = seed

    results = {}

    for value in range(10):#
        seed = time.time_ns() % (2**32)
        print(f"Seed is: {seed}")
        args['seed'] = seed
#        args['term_reg'] = test_values[value]
#        args["num_options"] = value
#        args["noise_std"] = test_values[value]
#        print(f"Value: {test_values[value]}")
    
        lens, steps, shortest_path, shortest_ops, shortest_path_time, shortest_ol_ratio, shortest_op_lengths, convergence_time, options_lengths_ave, ave_op_length, op_len_ratio, short_op_len_ratio, op_terms, greedy_ops, max_lengths = run(args, env)
        results[f"op_{value}"] = {
            "lens": lens, "steps": steps, "shortest_path": shortest_path, "shortest_ops": shortest_ops, "shortest_path_time": shortest_path_time,
            "shortest_ol_ratio": shortest_ol_ratio, "shortest_op_lengths": shortest_op_lengths,
            "convergence_time": convergence_time, "options_lengths_ave": options_lengths_ave, "ave_op_length": ave_op_length,
            "op_len_ratio": op_len_ratio, "short_op_len_ratio": short_op_len_ratio, "op_terms": op_terms, "greedy_ops": greedy_ops, "max_lengths": max_lengths,
        }
        print(f"Test finished")

    with open(f"/home/b/brendon45/oc_tests/data/{experiment_name}_{seed}.pkl", 'wb') as fp:
        pickle.dump(results, fp)



#    seed = time.time_ns() % (2**32)
#    print(f"Seed is {seed}")
#    args['seed'] = seed

#    term_test_values = [0.00, 0.01, 0.03, 0.05, 0.07, 0.09]
#    options_test = [2, 3, 4, 5, 6]

#    horizon = args["horizon"]

#    maze_file = f"/home/b/brendon45/oc_tests/mazes/{args['maze_name']}.xml"
#    file_name = f"/home/b/brendon45/oc_tests/pc_grids/{args['pc_file']}"
#    env = FAIRISEnvTF(maze_file, horizon, file_name, False, 1)

#    print("Print termination experiment")
#    print(f"Tested values: {term_test_values}")
#    print("Print num options experiment")
#    print(f"Tested values: {options_test}")

#    results = {}

#    for value in term_test_values:#
#        print(f"Current test value: {value}")
#        args["term_reg"] = value
#        args["num_options"] = value
#        lens, steps, shortest_path, shortest_ops, shortest_path_time, shortest_ol_ratio, shortest_op_lengths, convergence_time, options_lengths_ave, ave_op_length, op_len_ratio, short_op_len_ratio, op_terms, greedy_ops, max_lengths = run(args, env)
#        results[f"op_{value}"] = {
#            "lens": lens, "steps": steps, "shortest_path": shortest_path, "shortest_ops": shortest_ops, "shortest_path_time": shortest_path_time,
#            "shortest_ol_ratio": shortest_ol_ratio, "shortest_op_lengths": shortest_op_lengths,
#            "convergence_time": convergence_time, "options_lengths_ave": options_lengths_ave, "ave_op_length": ave_op_length,
#            "op_len_ratio": op_len_ratio, "short_op_len_ratio": short_op_len_ratio, "op_terms": op_terms, "greedy_ops": greedy_ops, "max_lengths": max_lengths,
#        }
#        print(f"Test finished")

#    with open(f"/home/b/brendon45/oc_tests/data/term_experiment_results_{seed}.pkl", 'wb') as fp:
#        pickle.dump(results, fp)

