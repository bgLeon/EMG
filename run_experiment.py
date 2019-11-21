import gym, argparse, time
import tensorflow as tf
from algorithms import i_dqn_l, i_lpopl
from utils.utils import Tester, Saver, TestingParameters
from utils.curriculum import CurriculumLearner

class TrainingParameters:
	def __init__(self, final_lr=0.0005, max_timesteps_per_spec=25000,
				replay_size=25000, print_freq=1000, exploration_frac=0.2, 
				final_exploration=0.02, values_network_update_freq=1,
				batch_size=32, learning_starts=1000, gamma=0.9,
				target_network_update_freq=100):
		"""Parameters
		-------
		lr: float
			final learning rate
		max_timesteps_per_spec: int
			max number of env steps for optimizer
		replay_size: int
			size of the replay buffer
		exploration_frac: float
			fraction of whole learning process while the exploration chance decays
		final_exploration: float
			final value of exploration
		model_update_freq: int
			the model is updated every `update_freq` steps.
		batch_size: int
			size of a batch from the replay buffer used for training
		print_freq: int
			how often to print out training progress
			set to None to disable printing
		learning_starts: int
			how many steps should the model collect transitions before start
			learning
		gamma: float
			discount factor for future rewards
		target_network_update_freq: int
			update the target network every `target_network_update_freq` steps.
		"""
		self.final_lr = final_lr
		self.max_timesteps_per_spec = max_timesteps_per_spec
		self.replay_size = replay_size
		self.exploration_frac = exploration_frac
		self.final_exploration = final_exploration
		self.values_network_update_freq = values_network_update_freq
		self.batch_size = batch_size
		self.print_freq = print_freq
		self.learning_starts = learning_starts
		self.gamma = gamma
		self.target_network_update_freq = target_network_update_freq

def run_experiment(alg_name, map_id, specs_id, num_times, steps, r_good,
	multi_ag, show_print, render):
	# configuration of testing params
	testing_params = TestingParameters()

	# configuration of learning params
	training_params = TrainingParameters()

	# Setting the experiment
	tester = Tester(training_params, testing_params, map_id, specs_id, multi_ag)

	# Setting the curriculum learner
	curriculum = CurriculumLearner(tester.specs, total_steps = steps,
									r_good = r_good)

	# Setting up the saver
	saver = Saver(alg_name, tester, curriculum)

	# Baseline 1 (Decentrlized LTL DQN)
	if alg_name == "i-dqn-l":
		i_dqn_l.run_experiments(tester, curriculum, saver, num_times,
			show_print, render)
	else:

		i_lpopl.run_experiments(tester, curriculum, saver, num_times, 
								show_print, render)
def run_multiple_experiments(alg, num_steps, specifications_id, multi_ag,
								show_print, render):
	num_times = 3
	num_maps = 1
	show_print = True
	r_good	 = 0.5 if specifications_id == 2 else 0.98
	time_init = time.time()
	for map_id in range(num_maps):
		print("Running", "r_good:", r_good, "alg:", alg, "map_id:", map_id,
			"specifications_id:", specifications_id)
		run_experiment(alg, map_id, specifications_id, num_times, num_steps,
						r_good, multi_ag, show_print, render)
	print("Total time:", "%0.2f"%((time.time() - time_init)/60), "mins")
	print("Avg time per run:", "%0.2f"%((time.time() - time_init)/60*num_times),
										"mins")
	print("Avg time per map:", "%0.2f"%((time.time() - time_init)/(60*num_maps\
										*num_times)), "mins")

# not in use for now
def run_single_experiment(alg, num_steps, specifications_id, map_id, multi_ag,
							show_print, render):
	num_times  = 1
	r_good = 0.5 if specifications_id == 2 else 0.98

	if show_print: print("Running", "r_good:", r_good, "alg:",
		alg, "map_id:", map_id, "num_steps:", num_steps,
		"specifications_id:", specifications_id)
	run_experiment(alg, map_id, specifications_id, num_times, num_steps,
					r_good, multi_ag, show_print, render)


if __name__ == "__main__":

	algorithms = ["i-dqn-l", "i-lpopl"]
	specifications = ["sequence", "interleaving", "safety"]

	parser = argparse.ArgumentParser(prog="run_experiments",
		description='Runs a multi-specification RL experiment over a gridworld\
		 	domain that is inspired on Minecraft.')
	parser.add_argument('--num_steps', default = 250000, type=int,
		help='total number of training steps')
	parser.add_argument('--show_Print', default = True, 
		action ='store_false', help='this paremeter tells if print progress')
	parser.add_argument('--render', default = False, 
		action ='store_true', help='this paremeter tells if the map is rendered')
	parser.add_argument('--singleA', default = False, 
		action ='store_true', help='this paremeter selects single agent \
									experiments, multi-agent are default opt')
	parser.add_argument('--algorithm', default='i-dqn-l', type=str, 
						help='This parameter indicated which RL algorithm to \
						use. The options are: ' + str(algorithms))
	parser.add_argument('--specifications', default='interleaving', type=str, 
		help='This parameter indicated which specifications to solve. \
		The options are: ' + str(specifications))
	"""
	map arg disabled until more maps are added
	"""
	# parser.add_argument('--map', default=-1, type=int,
	# 	help='This parameter indicated which map to use. \
	# 	It must be a number between -1 and 9. Use "-1" to run \
	# 	experiments over the 10 maps, 3 times per map')
	args = parser.parse_args()

	if args.algorithm not in algorithms: raise NotImplementedError("Algorithm "\
		+ str(args.algorithm) + " hasn't been implemented yet")
	if args.specifications not in specifications: raise NotImplementedError(
		"specifications " + str(args.specifications) + \
		" hasn't been defined yet")
	# if not(-1 <= args.map < 10): raise NotImplementedError(
	# 	"The map must be a number between -1 and 9")
	if not (args.num_steps>0): raise NotImplementedError(
		"The number of steps should be a positive integer")

	spec_id = specifications.index(args.specifications)

	# if args.map > -1:
	# 	run_single_experiment(args.algorithm, int(args.num_steps),
	# 		spec_id, -1, args.multiA, args.show_Print, args.render)
	# else:
	run_multiple_experiments(args.algorithm, int(args.num_steps),
		spec_id, args.singleA, args.show_Print, args.render)