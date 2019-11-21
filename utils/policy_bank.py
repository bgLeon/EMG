import tensorflow as tf
import os.path, time
import numpy as np
from utils.network import get_MLP

"""
This class includes a list of policies (a.k.a neural nets) for achieving different LTL goals
"""
class PolicyBank:
    def __init__(self, sess, num_actions, num_features, learning_params):
        self.sess = sess
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_params = learning_params
        # Inputs to the network
        self.s1 = tf.placeholder(tf.float64, [None, num_features])
        self.a = tf.placeholder(tf.int32)
        self.s2 = tf.placeholder(tf.float64, [None, num_features])
        # List of policies
        self.policies = []
        self.policy2id = {}
        # adding 'False' and 'True' policies
        self._add_constant_policy("False", 0.0)
        self._add_constant_policy("True", 1/learning_params.gamma) # this ensures that reaching 'True' gives reward of 1

    def _add_constant_policy(self, ltl, value):
        policy = ConstantPolicy(ltl, value, self.s2, self.num_features)
        self._add_policy(ltl, policy)

    def add_LTL_policy(self, ltl, dfa):
        if ltl not in self.policy2id:
            policy = Policy(ltl, dfa, self.sess, self.s1, self.a, self.s2, 
                            self.num_features, self.num_actions,
                            self.learning_params.gamma, 
                            self.learning_params.final_lr)
            self._add_policy(ltl, policy)

    def get_id(self, ltl):
        return self.policy2id[ltl]

    def get_LTL_policies(self):
        return set(self.policy2id.keys()) - set(['True', 'False'])

    def _add_policy(self, ltl, policy):
        self.policy2id[ltl] = len(self.policies)
        self.policies.append(policy)

    def get_number_LTL_policies(self):
        return len(self.policies) - 2 # The '-2' is because of the two constant policies ('False' and 'True')

    def reconnect(self):
        # Redefining connections between the different DQN networks
        num_policies = self.get_number_LTL_policies() 
        self.next_goal = tf.placeholder(tf.int32, [None, num_policies])
        batch_size = tf.shape(self.next_goal)[0]
        
        # concatenating q_target of every policy
        Q_target_all = tf.concat([self.policies[i].get_q_target_value() for i in range(len(self.policies))], 1)

        # Indexing the right target using 'goal' 
        aux_range = tf.reshape(tf.range(batch_size),[-1,1])
        aux_ones = tf.ones([1, num_policies], tf.int32)
        delta = tf.matmul(aux_range * (num_policies+2), aux_ones) 
        Q_target_index = tf.reshape(self.next_goal+delta, [-1])
        Q_target_flat = tf.reshape(Q_target_all, [-1])
        self.Q_target = tf.reshape(tf.gather(Q_target_flat, Q_target_index),[-1,num_policies]) 
        # NOTE: Q_target is batch_size x num_policies tensor such that 
        #       Q_target[i,j] is the target Q-value for policy "j+2" in instance 'i'

    
    def learn(self, s1, a, s2, next_goal):
        # computing the q_target values per policy
        Q_target = self.sess.run(self.Q_target, {self.s2: s2, self.next_goal: next_goal})
        # training using the Q_target
        values = {self.s1: s1, self.a: a}
        train = []
        for i in range(self.get_number_LTL_policies()):
            p = self.policies[i+2]
            values[p.Q_target] = Q_target[:,i]
            train.append(p.train)
        self.sess.run(train, values)

    """    
    def getLoss(self, s1, a, s2, next_goal):
        # computing the q_target values per policy
        Q_target = self.sess.run(self.Q_target, {self.s2: s2, self.next_goal: next_goal})
        # training using the Q_target
        values = {self.s1: s1, self.a: a}
        loss = []
        for i in range(self.get_number_LTL_policies()):
            p = self.policies[i+2]
            values[p.Q_target] = Q_target[:,i]
            loss.append(p.loss)
        return sum(self.sess.run(loss, values))
    """

    def get_best_action(self, ltl, s1):
        return self.sess.run(self.policies[self.policy2id[ltl]].get_best_action(), {self.s1: s1})

    def update_target_network(self):
        for i in range(self.get_number_LTL_policies()):
            self.policies[i+2].update_target_network()
    
    def get_policy_next_LTL(self, ltl, true_props):
        return self.policies[self.get_id(ltl)].dfa.progress_LTL(ltl, true_props)


class ConstantPolicy:
    def __init__(self, ltl, value, s2, num_features):
        self.ltl, self.value = ltl, value
        self._initialize_model(value, s2, num_features)

    def _initialize_model(self, value, s2, num_features):
        W = tf.constant(0, shape=[num_features, 1], dtype=tf.float64)
        b = tf.constant(value, shape=[1], dtype=tf.float64)
        self.q_target_value = tf.matmul(s2, W) + b

    def get_q_target_value(self):
        # Returns a vector of 'value' 
        return self.q_target_value

class Policy:
    def __init__(self, ltl, dfa, sess, s1, a, s2, num_features, num_actions, gamma, lr):
        self.dfa, self.sess = dfa, sess
        self.ltl_scope_name = str(ltl).replace("&","AND").replace("|","OR").replace("!","NOT").replace("(","P1_").replace(")","_P2").replace("'","").replace(" ","").replace(",","_")
        self._initialize_model(s1, a, s2, num_features, num_actions, gamma, lr)

    def _initialize_model(self, s1, a, s2, num_features, num_actions, gamma, lr):
        num_neurons = 64
        num_hidden_layers = 2
        self.Q_target = tf.placeholder(tf.float64)

        with tf.variable_scope(self.ltl_scope_name): # helps to give different names to this variables for this network
            # Defining regular and target neural nets
            self.q_values, self.q_target, self.update_target = get_MLP(s1, s2, num_features, num_actions, num_neurons, num_hidden_layers)
            # Q_values -> get optimal actions
            self.best_action = tf.argmax(self.q_values, 1)
            # Q_target -> set target value 'r + gamma * max Q_t' (r = 0 because ltl != True )
            self.q_target_value = tf.reshape(gamma * tf.reduce_max(self.q_target, axis=1), [-1,1])
            
            # Optimizing with respect to 'self.Q_target'
            action_mask = tf.one_hot(indices=a, depth=num_actions, dtype=tf.float64)
            Q_current = tf.reduce_sum(tf.multiply(self.q_values, action_mask), 1)
            self.loss = 0.5 * tf.reduce_sum(tf.square(Q_current - self.Q_target))
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            self.train = optimizer.minimize(loss=self.loss)
            
            # Initializing the network values
            self.sess.run(tf.variables_initializer(self._get_network_variables()))
            self.update_target_network() #copying weights to target net


    def _get_network_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.ltl_scope_name)

    def update_target_network(self):
        self.sess.run(self.update_target)

    def get_best_action(self):
        return self.best_action

    def get_q_target_value(self):
        return self.q_target_value

