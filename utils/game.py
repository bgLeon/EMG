from utils.game_objects import *
from utils.dfa import *
import random, math, os, sys, copy
import numpy as np

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class GameParams:
    def __init__(self, file_map, ltl_spec, consider_night):
        self.file_map = file_map
        self.ltl_spec = ltl_spec
        self.consider_night = consider_night

class Game:

    def __init__(self, params):
        self.params = params
        self._load_map(params.file_map)
        # Adding day and night if need it
        self.consider_night = params.consider_night
        self.nSteps = 0
        self.hour = 12
        if self.consider_night:
            self.sunrise = 5
            self.sunset  = 21
        # Loading and progressing the LTL reward
        self.dfa = DFA(params.ltl_spec)
        reward, self.ltl_game_over, self.env_game_over = self._get_rewards()
        reward = 0 # for initialization we don't penalize the agents
        for agent in self.agents.values():
            agent.update_reward(reward)

    def execute_actions(self, actions):
        """
        We execute 'action' in the game
        Returns the reward that the agent gets after executing the action 
        """
        agents = self.agents
        self.hour = (self.hour + 1)%24

        # So that the agents do not always make an action in the same order
        r = list(range(self.n_agents))
        random.shuffle(r)
         # Getting new position after executing action
        for i in r:
            agent = agents[i]
            action = actions[i]

            # Getting new position after executing action
            i,j = agent.i,agent.j
            ni,nj = self._get_next_position(action, i, j)

            # Interacting with the objects that is in the next position
            action_succeeded = self.map_array[ni][nj].interact(agent)

            # Action can only fail if the new position is a wall
            # or another agent
            if action_succeeded:
                # changing agent position in the map
                self._update_agent_position(agent, ni, nj)

        # Progressing the LTL reward and dealing with the consequences...
        # as it is right now all the agents share reward and consequences
        reward, self.ltl_game_over, self.env_game_over = self._get_rewards()
        for agent in agents.values():
            agent.update_reward(reward)
        self.nSteps+=1
        # we continue playing
        return reward

    def _get_next_position(self, action, ni, nj):
        """
        Returns the position where the agent would be if we execute action
        """
            
        # OBS: Invalid actions behave as wait
        if action == Actions.up   : ni-=1
        if action == Actions.down : ni+=1
        if action == Actions.left : nj-=1
        if action == Actions.right: nj+=1
        
        return ni,nj

    # Updates the map with the new position of the agent
    def _update_agent_position(self, agent, ni, nj):

        i, j = agent.i, agent.j
        # we recover what was previously there
        self.map_array[i][j] = self.back_map[i][j] 
        agent.change_position(ni,nj)
        self.map_array[ni][nj] = agent # we update the map with the agent

    def get_actions(self, agent):
        """
        Returns the list with the actions that the given agent can perform
        """
        return agent.get_actions()

    def _get_rewards(self):
        """
        This method progress the dfa and returns the 'reward' 
        and if its game over
        """
        if not self.consider_night:
            reward = -1
            ltl_game_over = False
            env_game_over = False if self.nSteps < 300 else True
            # env_game_over = False
            for agent in self.agents.values():
                true_props = self.get_true_propositions(agent)
                progressed = self.dfa.progress(true_props)
                ltl_game_over = self.dfa.is_game_over() or ltl_game_over
                if progressed: reward = 0
                if self.dfa.in_terminal_state(): 
                    reward = 1
                    """
                    For now we allow only one dfa progression at the same time, 
                    This is in order to easily keep track of the optimal policies
                    """
                    # break 
            return reward, ltl_game_over, env_game_over
        else:
            reward = -1
            ltl_game_over = False
            env_game_over = False if self.nSteps < 500 else True
            for agent in self.agents.values():
                true_props = self.get_true_propositions(agent)
                progressed = self.dfa.progress(true_props)
                if progressed: reward = 0
                ltl_game_over = self.dfa.is_game_over() or ltl_game_over
                if ltl_game_over: reward = -1*(500-self.nSteps)
                if self.dfa.in_terminal_state(): 
                    reward = 1
                    """
                    For now we allow only one dfa progression at the same time, 
                    This is in order to easily keep track of the optimal 
                    policies
                    """
                    # break
            return reward, ltl_game_over, env_game_over

    def get_LTL_goal(self):
        """
        Returns the next LTL goal
        """
        return self.dfa.get_LTL()

    def _is_night(self):
        return not(self.sunrise <= self.hour <= self.sunset)

    def _steps_before_dark(self):
        if self.sunrise - 1 <= self.hour <= self.sunset:
            return 1 + self.sunset - self.hour
        return 0 # it is night

    """
    Returns the string with the propositions that are True in this state
    """
    def get_true_propositions(self, agent):
        ret = str(self.back_map[agent.i][agent.j]).strip()
        # adding the is_night proposition
        if self.consider_night and self._is_night():
            ret += "n"
        return ret
    """
    The following methods return a feature representations of the map ------------
    for a given agent
    """
    def get_observation(self, agent):
        # map from object classes to numbers
        class_ids = self.class_ids #{"a":0,"b":1}
        N,M = self.map_height, self.map_width
        ret = []
        for i in range(N):
            for j in range(M):
                obj = self.map_array[i][j]
                if str(obj) in class_ids:
                    ret.append(self._manhattan_distance(obj, agent))
                # If we remove the lines below agents would oclude objects in 
                # their position to all agents
                obj2 = self.back_map[i][j]
                if str(obj) != str(obj2) and str(obj2) in class_ids:
                        ret.append(self._manhattan_distance(obj2, agent))
        
        # Adding the number of steps before night (if need it)
        if self.consider_night:
            ret.append(self._steps_before_dark())

        return np.array(ret, dtype=np.float64)


    def _manhattan_distance(self, obj, agent):
        """
        Returns the Manhattan distance between 'obj' and the agent
        """
        return abs(obj.i - agent.i) + abs(obj.j - agent.j)


    # The following methods create a string representation of the current state ---------
    def show_map(self):
        """
        Prints the current map
        """
        if self.consider_night:
            print(self.__str__(),"\n",
                "Steps before night:", self._steps_before_dark(),
                "Current time:", self.hour,
                "\n" ,"Reward:", self.agents[0].reward,
                "Agent has", self.agents[0].num_keys, "keys.", 
                "Goal", self.get_LTL_goal())
        else:
            print(self.__str__(),"\n", "Reward:", self.agents[0].reward,
                "Agent has", self.agents[0].num_keys, "keys.", "Goal",
                self.get_LTL_goal())
            # print("Steps before night:", self._steps_before_dark(), "Current time:", self.hour)
        # print("Reward:", self.agent.reward, "Agent has", self.agent.num_keys, "keys.", "Goal", self.get_LTL_goal())

    def __str__(self):
        return self._get_map_str()

    def _get_map_str(self):
        r = ""
        agent = self.agents[0]
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if agent.idem_position(i,j):
                    s += str(agent)
                else:
                    s += str(self.map_array[i][j])
            if(i > 0):
                r += "\n"
            r += s
        return r



    # The following methods create the map ----------------------------------------------
    def _load_map(self,file_map):
        """
        This method adds the following attributes to the game:
            - self.map_array: array containing all the static objects in the map
                - e.g. self.map_array[i][j]: contains the object located on row 
                        'i' and column 'j'
            - self.agents: are the agents
            - self.map_height: number of rows in every room 
            - self.map_width: number of columns in every room
        The inputs:
            - file_map: path to the map file
        """
        # contains all the actions that the agent can perform
        actions = self._load_actions()
        # loading the map
        self.map_array = []
        # I use the lower case letters to define the features
        self.class_ids = {} 
        self.agents = {}
        f = open(file_map)
        i,j = 0,0
        ag = 0
        for l in f:
            # I don't consider empty lines!
            if(len(l.rstrip()) == 0): continue
            # this is not an empty line!
            row = []
            b_row = []
            j = 0
            for e in l.rstrip():
                if e in "abcdefghijklmnopqrstuvwxyzH":
                    entity = Empty(i,j,label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                # we need to declare the initial positions of agents 
                # to be potentially empty espaces (after they moved)
                elif e in " A": entity = Empty(i,j)
                elif e == "X":  entity = Obstacle(i,j)
                else:
                    raise ValueError('Unkown entity ', e)
                if e == "A":
                    self.agents[ag] = Agent(i,j,actions)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                    ag+=1
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1

        """
        We use this back map to check what was there when an agent leaves a 
        position
        """
        self.back_map = copy.deepcopy(self.map_array) 
        for agent in self.agents.values():
            i,j = agent.i, agent.j
            self.map_array[i][j] = agent
        f.close()
        # height width
        self.map_height, self.map_width = len(self.map_array),\
            len(self.map_array[0])
        self.n_agents = len(self.agents)
        # print("There are", self.n_agents, "agents")

    def _load_actions(self):
        return [Actions.up, Actions.right, Actions.down, Actions.left,
            Actions.wait]


def play(params, max_time):
    # commands
    str_to_action = {"w":Actions.up,"d":Actions.right,"s":Actions.down,
        "a":Actions.left, "e":Actions.wait}
    # play the game!
    game = Game(params)
    for t in range(max_time):
        # Showing game
        game.show_map()
        acts = game.get_actions(game.agents[0])
        # Getting action
        print("\nSteps ", t)
        print("Action? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action and str_to_action[a] in acts:
            aactions = [str_to_action[a], Actions.up, Actions.up]
            reward = game.execute_actions(aactions)
            if game.ltl_game_over or game.env_game_over: # Game Over
                break 
        else:
            print("Forbidden action")
    game.show_map()
    return reward


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    import specifications
    map = "./debug/multiA_map_0.txt"
    specs = specifications.get_safety_constraints()
    max_time = 100
    consider_night=True

    for t in specs:
        t = specs[-1]
        while True:
            params = GameParams(map, t, consider_night)
            if play(params, max_time) > 0:
                break
