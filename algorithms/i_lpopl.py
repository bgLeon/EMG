# Imports
import numpy as np
import random, time, shutil, os.path
import tensorflow as tf
from utils.policy_bank import *
from utils.schedules import LinearSchedule
from utils.replay_buffer import ReplayBuffer
from utils.dfa import *
from utils.game import *

	
def _run_ILPOPL(sess, policy_banks, spec_params, tester, curriculum,
				show_print, render):
	# Initializing parameters
	training_params = tester.training_params
	testing_params = tester.testing_params
	
	# Initializing the game
	env = Game(spec_params)
	agents = env.agents
	action_set = env.get_actions(agents[0])

	# Initializing experience replay buffers
	replay_buffers = {}
	for agent in range(env.n_agents):
		replay_buffers[str(agent)] = ReplayBuffer(training_params.replay_size)
		
	# Initializing parameters
	num_features = len(env.get_observation(agents[0]))
	max_steps = training_params.max_timesteps_per_spec
	exploration = LinearSchedule(schedule_timesteps = int(
								training_params.exploration_frac * max_steps),
								initial_p = 1.0,
	
								final_p = training_params.final_exploration)
	last_ep_rew = 0	
	training_reward = 0
	episode_count = 0 
	# Starting interaction with the environment
	if show_print: print("Executing", max_steps, "actions...")
	if render: env.show_map()

	#We start iterating with the environment
	for t in range(max_steps):
		# Getting the current state and ltl goal
		actions = []
		ltl_goal = env.get_LTL_goal()
		for agent, policy_bank in zip(agents.values(), policy_banks.values()):
			s1 = env.get_observation(agent)

			# Choosing an action to perform
			if random.random() < exploration.value(t): 
				act = random.choice(action_set)
			else: act = Actions(policy_bank.get_best_action(ltl_goal, 
												s1.reshape((1,num_features))))
			actions.append(act)
		# updating the curriculum
		curriculum.add_step()
				
		# Executing the action
		reward = env.execute_actions(actions)
		training_reward += reward

		if render and episode_count%30 is 0:
			time.sleep(0.01)
			clear_screen()
			env.show_map()

		true_props = []
		for agent in agents.values():
			true_props.append(env.get_true_propositions(agent))
		# Saving this transition
		for agent, policy_bank, replay_buffer, act in zip(agents.values(), 
													policy_banks.values(),
													replay_buffers.values(),
													actions):
			s2 = env.get_observation(agent)
			next_goals = np.zeros((policy_bank.get_number_LTL_policies(),),
									dtype=np.float64)
			for ltl in policy_bank.get_LTL_policies():
				ltl_id = policy_bank.get_id(ltl)
				if env.env_game_over:
					# env deadends are equal to achive the 'False' formula
					ltl_next_id = policy_bank.get_id("False") 
				else:
					for props in true_props:
						ltl_next_id = policy_bank.get_id(\
								policy_bank.get_policy_next_LTL(ltl, props))
				next_goals[ltl_id-2] = ltl_next_id
			replay_buffer.add(s1, act.value, s2, next_goals)
		
			# Learning
			if curriculum.get_current_step() > training_params.learning_starts\
				and curriculum.get_current_step() %\
				training_params.values_network_update_freq == 0:
				# Minimize the error in Bellman's equation on a batch sampled 
				# from replay buffer.
				S1, A, S2, Goal = replay_buffer.sample(
													training_params.batch_size)
				policy_bank.learn(S1, A, S2, Goal)
			
			# Updating the target network
			if curriculum.get_current_step() > training_params.learning_starts\
				and curriculum.get_current_step() %\
				training_params.target_network_update_freq == 0:
				# Update target network periodically.
				policy_bank.update_target_network()

		# Printing
		if show_print and (curriculum.get_current_step()+1) \
							% training_params.print_freq == 0:
			print("Step:", curriculum.get_current_step()+1,
				"\tLast episode reward:", last_ep_rew, "\tSucc rate:",
				"%0.3f"%curriculum.get_succ_rate(),
				"\tNumber of episodes:", episode_count)

		# Testing
		if testing_params.test and curriculum.get_current_step() %\
												testing_params.test_freq == 0:
			tester.run_test(curriculum.get_current_step(), sess,
							_test_ILPOPL, policy_banks, num_features)

		# Restarting the environment (Game Over)
		if env.ltl_game_over or env.env_game_over:
			# NOTE: Game over occurs for one of three reasons: 
			# 1) DFA reached a terminal state, 
			# 2) DFA reached a deadend, or 
			# 3) The agent reached an environment deadend (e.g. a PIT)
			env = Game(spec_params) # Restarting
			agents = env.agents
			episode_count+=1
			last_ep_rew = training_reward

			training_reward = 0
			# updating the hit rates
			curriculum.update_succ_rate(t, reward)
			# Uncomment if want to stop learning according to succ. rate
			# if curriculum.stop_spec(t):
			# 	last_ep_rew = 0
			# 	if show_print: print("STOP SPEC!!!")
			# 	break
		
		# checking the steps time-out
		if curriculum.stop_learning():
			if show_print: print("STOP LEARNING!!!")
			break

	if show_print: 
		print("Done! Last reward:", last_ep_rew)


def _test_ILPOPL(sess, spec_params, training_params, testing_params,
				policy_banks, num_features):
	# Initializing parameters
	env = Game(spec_params)
	agents = env.agents
	# Starting interaction with the environment
	r_total = 0
	for t in range(testing_params.num_steps):
		# Getting the current state and ltl goa
		actions = []
		for agent, policy_bank in zip(agents.values(), policy_banks.values()):
			s1 = env.get_observation(agent)

			# Choosing an action to perform
			act = Actions(policy_bank.get_best_action(env.get_LTL_goal(),
												s1.reshape((1,num_features))))
			actions.append(act)
			# Executing the action
		r_total += env.execute_actions(actions) * training_params.gamma**t

		# Restarting the environment (Game Over)
		if env.ltl_game_over or env.env_game_over:
			break
	return r_total

def _initialize_policy_banks(sess, training_params, curriculum, tester):
	policy_banks = {}
	env = Game(tester.get_spec_params(curriculum.get_current_spec()))
	# For now we assume all agents have the same input and output dimensions
	num_features = len(env.get_observation(env.agents[0]))
	num_actions  = len(env.get_actions(env.agents[0]))
	for agent in range(env.n_agents):
		policy_banks[str(agent)] = PolicyBank(sess, num_actions, num_features,
												training_params)
		for f_task in tester.get_LTL_specs():
			dfa = DFA(f_task)
			for ltl in dfa.ltl2state:
				# this method already checks that the policy is not in the bank 
				# and it is not 'True' or 'False'
				policy_banks[str(agent)].add_LTL_policy(ltl, dfa)
		policy_banks[str(agent)].reconnect() # -> creating the connections 
												# between the neural nets
	  
	print("\n", policy_banks["0"].get_number_LTL_policies(), 
		"sub-tasks per agent were extracted!\n")
	return policy_banks

def run_experiments(tester, curriculum, saver, num_times, show_print, render):
	# Running the tasks 'num_times'
	time_init = time.time()
	training_params = tester.training_params
	for t in range(num_times):
		# Setting the random seed to 't'
		random.seed(t)
		tf.set_random_seed(t)
		tf_config = tf.ConfigProto()
		tf_config.inter_op_parallelism_threads=2
		tf_config.intra_op_parallelism_threads=2
		tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
		sess = tf.Session(config=tf_config, graph=None)
		
		# Reseting default values
		curriculum.restart()
		
		# Initializing policies per each subtask
		policy_banks = _initialize_policy_banks(sess, training_params,
												curriculum, tester)

		# Running the tasks
		while not curriculum.stop_learning():
			if show_print: print("Current step:",
								curriculum.get_current_step(), "from",
								curriculum.total_steps)
			spec = curriculum.get_next_spec()
			spec_params = tester.get_spec_params(spec)
			_run_ILPOPL(sess, policy_banks, spec_params, tester, 
						curriculum, show_print, render)
		
		tf.reset_default_graph()
		sess.close()
		
		# Backing up the results
		saver.save_results()

	# Showing results
	tester.show_results()
	print("Time:", "%0.2f"%((time.time() - time_init)/60), "mins")

