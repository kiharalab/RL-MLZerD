# MIT License
# Copyright (c) 2018 Valentyn N Sichkar (Valentyn N Sichkar. Reinforcement Learning Algorithms for global path planning // GitHub platform. DOI: 10.5281/zenodo.1317899)
# github.com/sichkar-valentyn
# For skeleton functions
#
# MIT License
# Copyright (c) 2021 Tunde Aderinwale, Daisuke Kihara, and Purdue University 
# For function implementations


# Importing classes
from env import Environment
from qtable import QLearningTable
from argparser import argparser
import logging
import time
import sys
import os
import numpy as np

def update(no_of_episodes,qtable_path,qtable_out,log):
    # Resulted list for the plotting Episodes via Steps
    steps = []

    # Summed costs for all episodes in resulted list
    all_costs = 0
    last_total = 0
    total_reward = 0
    repeat = False

    for episode in range(no_of_episodes):
        # Initial Observation
        #print("Currently running episode:",episode)
        observation = env.reset()
        transition_tracker = {}
        # Updating number of Steps for each Episode
        i = 0

        # Updating the cost for each episode
        cost = 0

        while True:
            # RL chooses action based on observation
            if repeat:
                action = RL.get_top5_actions(str(observation))
            else:
                action = RL.choose_action(str(observation)) # selects best decoy or random decoy base on chance

            # RL takes an action and get the next observation and reward
            observation_, reward, done , r, old_action = env.step(action,episode) # go to the next state, explore children
            repeat = r
            
            # Don't learn or change/track state changes
            if repeat:
                pass
                #action = RL.get_top5_actions(str(observation))
            
            else:
                # RL learns from this transition and calculating the cost
                if type(action) is list:
                    action = old_action
                cost += RL.learn(str(observation), action, reward, str(observation_),episode)

                #Keep track of transition: this state, next state and action that took us to next state
                if observation != None:
                    current_state = observation.copy()
                else:
                    current_state = None
                if observation_ not in ['goal','forest']:
                    next_state = observation_.copy()
                else:
                    next_state = observation_

                #store current and next state to be used for later
                if i == 0:
                    transition_tracker[i] = [current_state,next_state,action]
                else:
                    transition_tracker[i] = [transition_tracker[i-1][1],next_state,action] #current state is the previous next from last iteration

                # Swapping the observations - current and next
                observation = observation_

                # Calculating number of Steps in the current Episode
                i += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                steps += [i]
                all_costs += cost
                RL.backpropagate_reward(transition_tracker,reward)
                env.epsdecay()

                if env.refresh():
                    RL.reset_qtable()
                    log.info('Tree and Qtable was restarted/refreshed at episode : ' + str(episode))

                # write qtable snapshot if full reward is return
                #if reward == 100:
                #    RL.print_intermediate_q_table(qtable_out,episode)
                break
        #print()
        total_reward += reward # Keep score
        if episode % 1000 == 0: # Print out metadata every 250th iteration
            performance = (total_reward - last_total) / 1000.0
            last_total = total_reward
            print('episode: ', episode, 'performance: ', performance, 'total_reward: ', total_reward, 'all_costs: ', all_costs)
            log.info('episode: ' + str(episode) + ' performance: '+ str(performance) +' total_reward: '+ str(total_reward) + ' all_costs: ' + str(all_costs))

    # Showing the final route
    final_path = env.final()
    log.info('Docking Order : ' + final_path)
    print('Docking Order : ' + final_path)

    energy_stats = env.print_energy_stats()
    log.info('Total number of low energies discovered : ' + str(len(energy_stats)))
    log.info('List of low energies discovered : ' + str(energy_stats))
    env.print_metro_stats()

    # Showing the Q-table with values for each action
    RL.print_q_table(qtable_path)
    #log.info('LR Decay : ' + str(RL.print_lr_decay()))


# Commands to be implemented after running this file
if __name__ == "__main__":
    #parse Parameters
    params = argparser()

    protein = params['protein']
    no_of_chains = params['nofchains']
    chains = list(params['chains'])
    clash_thres = params['clash_threshold']
    pair_clash_thres = params['pair_clash_threshold']
    met_thres = params['metro_threshold']
    episode_no = params['episodes']
    epsilon = params['eps']
    lr = params['lr']
    data_dir = params['path']
    decoy_size = params['pool_size']
    out_dir = params['out_dir']
    not_int_pair = params['not_int_pair']
    int_pair = params['int_pair']
    term_thres = params['terminal_thres']
    unbound = params['unbound']
    fit_classifier = params['classifier']

    #Shouldn't change anything below this line...
    #____________________________________________________________________

    #prep outdirectory
    unbound_path = '/unbound' if unbound == 'yes' else ''
    output_dir = data_dir + '/data/' + protein + unbound_path + '/' + out_dir + '/'
    decoy_out = output_dir + 'output/'
    qtable_out = output_dir + '/qtables/'
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(decoy_out):
            os.makedirs(decoy_out)

        if not os.path.exists(qtable_out):
            os.makedirs(qtable_out)
    
    except Exception as e:
        print(e)


    logname = output_dir + protein + '_' + str(lr) + '_mcrl.out'
    qtable_path = output_dir + protein + '_' + str(lr) + '_qtable.csv'
    logging.basicConfig(filename=logname,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

    logging.info("Starting Multiple Docking for " + protein + " chain_list : " + str(chains))
    print("Starting Multiple Docking for : ", protein)
    start_time = time.time()
    logging.info("Docking Parameters: Clash Threshold = " + str(clash_thres) + " Pair Clash Threshold : " + str(pair_clash_thres) + " Total Episodes : " + str(episode_no))
    logging.info("Non Interacting Pair information : " + str(not_int_pair))
    logging.info("Interacting Pair information : " + str(int_pair))
    logging.info("Classifier used for Docking : " + fit_classifier)
    
    mylogger = logging.getLogger("MCRLLogger")

    # Calling for the environment
    env = Environment(chain_length=no_of_chains,chain_list=chains,protein_name=protein,thresh=clash_thres,pair_thres=pair_clash_thres,metro_thres=met_thres, epsln=epsilon, log=mylogger, learning_rate = lr, path=data_dir,outpath=output_dir,noiter=episode_no,nip=not_int_pair,ip=int_pair,term_thres=term_thres,unbound=unbound,ds=decoy_size,f_cla=fit_classifier)
    chains_energy = env.energy_stats()
    RL = QLearningTable(actions=list(range(decoy_size)),e_greedy=epsilon,iterations=episode_no,log=mylogger,learning_rate = lr,energy_details = chains_energy)
    # Running the main loop with Episodes by calling the function update()
    update(episode_no,qtable_path,qtable_out,mylogger)
    total_time = time.time() - start_time
    logging.info("Docking for  " + str(episode_no) + " Completed, Total running time : " + str(total_time))
    print("Docking for  " + str(episode_no) + " Completed, Total running time : " + str(total_time))
