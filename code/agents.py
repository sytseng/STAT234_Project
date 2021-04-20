"""
This file contains implementation of all the agents.
"""

from abc import ABC, abstractmethod
from util import *
import random
from game import CHECKERS_FEATURE_COUNT, checkers_features_augmented, checkers_features_simple, checkers_reward
from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam

class Agent(ABC):

    def __init__(self, is_learning_agent=False):
        self.is_learning_agent = is_learning_agent
        self.has_been_learning_agent = is_learning_agent

    @abstractmethod
    def get_action(self, state):
        """
        state: the state in which to take action
        Returns: the single action to take in this state
        """
        pass


class KeyBoardAgent(Agent):

    def __init__(self):
        Agent.__init__(self)


    def get_action(self, state):
        """
        state: the current state from which to take action

        Returns: list of starting position, ending position
        """

        start = [int(pos) for pos in input("Enter start position (e.g. x y): ").split(" ")]
        end = [int(pos) for pos in input("Enter end position (e.g. x y): ").split(" ")]

        ends = []
        i=1
        while i < len(end):
            ends.append([end[i-1], end[i]])
            i += 2

        action = [start] + ends
        return action

class RandomAgent(Agent):
    
    def __init__(self):
        Agent.__init__(self)
        
    def get_action(self, state):
        possible_moves = state.board.get_possible_next_moves()
        action = possible_moves[np.random.choice(len(possible_moves),size=1)[0]]
        return action
    

class AlphaBetaAgent(Agent):

    def __init__(self, depth, reward_direction=1):
        Agent.__init__(self, is_learning_agent=False)
        self.depth = depth
        self.reward_direction = reward_direction

    def evaluation_function(self, state, agent=True):
        """
        state: the state to evaluate
        agent: True if the evaluation function is in favor of the first agent and false if
               evaluation function is in favor of second agent

        Returns: the value of evaluation
        """
        agent_ind = 0 if agent else 1
        other_ind = 1 - agent_ind

        if state.is_game_over():
            if agent and state.is_first_agent_win():
                return 500*self.reward_direction

            if not agent and state.is_second_agent_win():
                return 500*self.reward_direction

            return -500*self.reward_direction

        pieces_and_kings = state.get_pieces_and_kings()
        return (pieces_and_kings[agent_ind] + 2 * pieces_and_kings[agent_ind + 2] - \
        (pieces_and_kings[other_ind] + 2 * pieces_and_kings[other_ind + 2]))*self.reward_direction

    def get_action(self, state):

        def mini_max(state, depth, agent, A, B):
            if agent >= state.get_num_agents():
                agent = 0

            depth += 1
            if depth == self.depth or state.is_game_over():
                return [None, self.evaluation_function(state, max_agent)]
            elif agent == 0:
                return maximum(state, depth, agent, A, B)
            else:
                return minimum(state, depth, agent, A, B)

        def maximum(state, depth, agent, A, B):
            output = [None, -float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent + 1, A, B)

                check = val[1]

                if check > output[1]:
                    output = [action, check]

                if check > B:
                    return [action, check]

                A = max(A, check)

            return output

        def minimum(state, depth, agent, A, B):
            output = [None, float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent+1, A, B)

                check = val[1]

                if check < output[1]:
                    output = [action, check]

                if check < A:
                    return [action, check]

                B = min(B, check)

            return output

        # max_agent is true meaning it is the turn of first player at the state in 
        # which to choose the action
        max_agent = state.is_first_agent_turn()
        output = mini_max(state, -1, 0, -float("inf"), float("inf"))
        return output[0]


class ReinforcementLearningAgent(Agent):

    def __init__(self, is_learning_agent=True, reward_function=checkers_reward):
        Agent.__init__(self, is_learning_agent)

        self.episodes_so_far = 0
        self.reward_function = reward_function


    @abstractmethod
    def get_action(self, state):
        """
        state: the current state from which to take action

        Returns: the action to perform
        """
        # TODO call do_action from this method
        pass


    @abstractmethod
    def update(self, state, action, next_state, reward):
        """
        performs update for the learning agent

        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        pass

    def start_episode(self):
        # Accumulate rewards while training for each episode and show total rewards 
        # at the end of each episode i.e. when stop episode
        self.prev_state = None
        self.prev_action = None

        self.episode_rewards = 0.0


    def stop_episode(self):
        # print('reward this episode', self.episode_rewards)
        pass

    @abstractmethod
    def start_learning(self):
        pass


    @abstractmethod
    def stop_learning(self):
        pass


    @abstractmethod
    def observe_transition(self, state, action, next_state, reward, next_action=None):
        pass


    @abstractmethod
    def observation_function(self, state):
        pass


#     # TODO
#     def reward_function(self, state, action, next_state):
#         # make a reward function for the environment
#         return checkers_reward(state, action, next_state)


    def do_action(self, state, action):
        """
        called by get_action to update previous state and action
        """
        self.prev_state = state
        self.prev_action = action


class QLearningAgent(ReinforcementLearningAgent):

    def __init__(self, feature_func=checkers_features_augmented, feature_count=34, reward_function=checkers_reward, alpha=0.01, gamma=0.1, epsilon=0.5, is_learning_agent=True, weights=None):

        """
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration constant
        is_learning_agent: whether to treat this agent as learning agent or not
        weights: default weights
        """

        ReinforcementLearningAgent.__init__(self, is_learning_agent=is_learning_agent, reward_function=reward_function)

        self.original_alpha = alpha
        self.original_epsilon = epsilon

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.feature_func = feature_func
        self.feature_count = feature_count

        if not is_learning_agent:
            self.epsilon = 0.0
            self.alpha = 0.0


        if weights is None:
            # initialize weights for the features
            self.weights = np.zeros(self.feature_count)
        else:
            if len(weights) != self.feature_count:
                raise Exception("Invalid weights " + weights)

            self.weights = np.array(weights, dtype=float)


    def start_learning(self):
        """
        called by environment to notify agent of starting new episode
        """

        self.alpha = self.original_alpha
        self.epsilon = self.original_epsilon

        self.is_learning_agent = True


    def stop_learning(self):
        """
        called by environment to notify agent about end of episode
        """
        self.alpha = 0.0
        self.epsilon = 0.0

        self.is_learning_agent = False


    def get_q_value(self, state, action, features):
        """
          Returns: Q(state,action)
        """
        q_value = np.dot(self.weights, features)
        return q_value


    def compute_value_from_q_values(self, state):
        """
          Returns: max_action Q(state, action) where the max is over legal actions.
                   If there are no legal actions, which is the case at the terminal state, 
                   return a value of 0.0.
        """
        actions = state.get_legal_actions()

        if not actions:
            return 0.0

        q_values = \
        [self.get_q_value(state, action, self.feature_func(state, action)) for action in actions]

        return max(q_values)


    def compute_action_from_q_values(self, state, actions):
        """
          Returns: the best action to take in a state. If there are no legal actions,
                   which is the case at the terminal state, return None.
        """
        if not actions:
            return None

        # if max_value < 0:
        #     return random.choice(actions)

        arg_max = np.argmax([self.get_q_value(state, action, self.feature_func(state, action)) 
            for action in actions])

        return actions[arg_max]


    def get_action(self, state):
        """
          Returns: the action to take in the current state.  With probability self.epsilon,
                   take a random action and take the best policy action otherwise.  If there are
                   no legal actions, which is the case at the terminal state, returns None.
        """

        # Pick Action
        legal_actions = state.get_legal_actions()
        action = None

        if not legal_actions:
            return None

        if flip_coin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_q_values(state, legal_actions)

        self.do_action(state, action)
        return action


    def update(self, state, action, next_state, reward):

        features = self.feature_func(state, action)

        expected = reward + self.gamma * self.compute_value_from_q_values(next_state)
        current = self.get_q_value(state, action, features)

        temporal_difference = expected - current

        for i in range(self.feature_count):
            self.weights[i] = self.weights[i] + self.alpha * (temporal_difference) * features[i]


    def getPolicy(self, state):
        return self.compute_action_from_q_values(state, state.get_legal_actions())


    def getValue(self, state):
        return self.compute_value_from_q_values(state)  


    def observe_transition(self, state, action, next_state, reward, next_action=None):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, reward)


    def observation_function(self, state):
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            # print('reward is', reward)
            self.observe_transition(self.prev_state, self.prev_action, state, reward)

    def update_parameters(self, freq, num_games):
        if num_games % freq == 0:
            self.original_alpha /= 2.0
            self.original_epsilon /= 2.0


class SarsaLearningAgent(QLearningAgent):

    def __init__(self, feature_func=checkers_features_augmented, feature_count=34, reward_function=checkers_reward, alpha=0.01, gamma=0.1, epsilon=0.5, is_learning_agent=True, weights=None):
        
        QLearningAgent.__init__(self, feature_func, feature_count,reward_function,alpha, gamma, epsilon, is_learning_agent, weights)
    
    def start_episode(self):
        # Accumulate rewards while training for each episode and show total rewards 
        # at the end of each episode i.e. when stop episode
        self.prev_state = None
        self.prev_action = None
        self.prev_prev_state = None
        self.prev_prev_action = None
        
        self.episode_rewards = 0.0

    def update(self, state, action, next_state, next_action, reward):

        features = self.feature_func(state, action)

        if next_action is None:
            next_q_value = 0.0
        else:
            next_q_value = \
            self.get_q_value(next_state, next_action, self.feature_func(next_state, next_action))
    
        expected = reward + self.gamma * next_q_value

        current = self.get_q_value(state, action, features)

        temporal_difference = expected - current

        for i in range(self.feature_count):
            self.weights[i] = self.weights[i] + self.alpha * (temporal_difference) * features[i]


    def observe_transition(self, state, action, next_state, next_action, reward):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, next_action, reward)


#     def observation_function(self, state):
#         if self.prev_state is not None:
#             reward = self.reward_function(self.prev_state, self.prev_action, state)
#             # print('reward is', reward)
#             action = self.get_action(state)
#             self.observe_transition(self.prev_state, self.prev_action, state, action, reward)

#             return action

    def observation_function(self, state):
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            self.prev_prev_state = self.prev_state
            self.prev_prev_action = self.prev_action
            action = self.get_action(state)
            self.observe_transition(self.prev_prev_state, self.prev_prev_action, self.prev_state, self.prev_action, reward)
 
            return action


# class SarsaSoftmaxAgent(SarsaLearningAgent):

#     def __init__(self, feature_func=checkers_features_augmented, feature_count=34, reward_function=checkers_reward, alpha=0.01, gamma=0.1, t=1.0, is_learning_agent=True, weights=None):
#         SarsaLearningAgent.__init__(self, feature_func=feature_func, feature_count=feature_count, reward_function=reward_function, alpha=alpha, gamma=gamma,
#             is_learning_agent=is_learning_agent, weights=weights)

#         self.t = t

#     def get_action(self, state):
#         legal_actions = state.get_legal_actions()

#         if not legal_actions:
#             return None

#         if self.epsilon == 0.0:
#             return self.compute_action_from_q_values(state, legal_actions)

#         q_values = [self.get_q_value(state, action, feature_func(state, action))
#                 for action in legal_actions]

#         exps = np.exp(q_values) / self.t
#         probs = exps / np.sum(exps)

#         action_ind = np.random.choice(len(legal_actions), p=probs)

#         self.do_action(state, legal_actions[action_ind])
#         return legal_actions[action_ind]

#     def update_parameters(self, freq, num_games):
#         if num_games % freq == 0:
#             self.t /= 2.0

class SarsaLambdaAgent(SarsaLearningAgent):
    
    def __init__(self, feature_func=checkers_features_augmented, feature_count=34, reward_function=checkers_reward, alpha=0.01, gamma=0.1, epsilon=0.5, lam=0.5, is_learning_agent=True, weights=None):
        SarsaLearningAgent.__init__(self, feature_func, feature_count, reward_function, alpha, gamma, epsilon, is_learning_agent, weights)
        
        self.lam = lam    
        
    def start_episode(self):
        # Accumulate rewards while training for each episode and show total rewards 
        # at the end of each episode i.e. when stop episode
        self.prev_state = None
        self.prev_action = None
        self.prev_prev_state = None
        self.prev_prev_action = None
        
        self.x = None
        self.Q_old = 0.0
        self.z = np.zeros((self.feature_count,1))
        
        self.episode_rewards = 0.0
        self.step_count = 0
        
    
    def update(self, state, action, next_state, next_action, reward):

        if self.x is None:
            self.x = np.array(self.feature_func(state, action)).reshape(-1,1)

        if next_action is None:
            x_prime = np.zeros_like(self.x)
        else:
            x_prime = np.array(self.feature_func(next_state, next_action)).reshape(-1,1)
            
        Q = self.get_q_value(state, action, self.x)
        Q_prime = self.get_q_value(next_state, next_action, x_prime)
        
        delta = reward + self.gamma * Q_prime - Q
        self.z = self.gamma * self.lam * self.z \
                 + (1. - self.alpha * self.gamma * self.lam * np.dot(self.z.T, self.x)) * self.x
        self.weights = (self.weights.reshape(-1,1) + self.alpha * (delta + Q - self.Q_old) * self.z \
                      - self.alpha * (Q - self.Q_old) * self.x).flatten()
        self.Q_old = Q_prime
        self.x = x_prime
        
        

    def observe_transition(self, state, action, next_state, next_action, reward):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, next_action, reward)


    def observation_function(self, state):
        self.step_count += 1
#         print(self.step_count)
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            self.prev_prev_state = self.prev_state
            self.prev_prev_action = self.prev_action
#             print('reward is', reward)
            action = self.get_action(state)

            self.observe_transition(self.prev_prev_state, self.prev_prev_action, self.prev_state, self.prev_action, reward)

            return action

class QLearningAgent_MLP(ReinforcementLearningAgent):

    def __init__(self, architecture, feature_func=checkers_features_augmented, feature_count=34, reward_function=checkers_reward, alpha=0.01, gamma=0.1, epsilon=0.5, is_learning_agent=True, weights=None):

        """
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration constant
        is_learning_agent: whether to treat this agent as learning agent or not
        weights: default weights
        """

        ReinforcementLearningAgent.__init__(self, is_learning_agent=is_learning_agent, reward_function=reward_function)

        self.original_alpha = alpha
        self.original_epsilon = epsilon

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = alpha
        
        self.feature_func = feature_func
        self.feature_count = feature_count
        self.architecture = architecture
        self.architecture['input_dim'] = feature_count
        
        self.params = {
            'step_size':1e-3,
            'max_iteration':1,
            'random_restarts':1,
            'reg_param':100.}

        if not is_learning_agent:
            self.epsilon = 0.0
            self.alpha = 0.0


        if weights is None:
            # initialize weights for the features
            self.nn = MLP(architecture)
        else:
            self.nn = MLP(architecture, weights = weights)

        self.mu = np.zeros(self.feature_count)

    def start_learning(self):
        """
        called by environment to notify agent of starting new episode
        """

        self.alpha = self.original_alpha
        self.epsilon = self.original_epsilon

        self.is_learning_agent = True


    def stop_learning(self):
        """
        called by environment to notify agent about end of episode
        """
        self.alpha = 0.0
        self.epsilon = 0.0

        self.is_learning_agent = False


    def get_q_value(self, state, action):
        """
          Returns: Q(state,action)
        """
        features = np.array(self.feature_func(state, action)).reshape(-1,1)
        q_value = self.nn.forward(self.nn.weights, features).flatten()
#         print(q_value, q_value.shape)
        return q_value


    def grad_q_value(self, features):
        
        def q_value(w):
            return self.nn.forward(w,features)
        
        return grad(q_value)
    

    def compute_value_from_q_values(self, state):
        """
          Returns: max_action Q(state, action) where the max is over legal actions.
                   If there are no legal actions, which is the case at the terminal state, 
                   return a value of 0.0.
        """
        actions = state.get_legal_actions()

        if not actions:
            return 0.0

        q_values = \
        [self.get_q_value(state, action) for action in actions]

        return max(q_values)


    def compute_action_from_q_values(self, state, actions):
        """
          Returns: the best action to take in a state. If there are no legal actions,
                   which is the case at the terminal state, return None.
        """
        if not actions:
            return None

        # if max_value < 0:
        #     return random.choice(actions)

        arg_max = np.argmax([self.get_q_value(state, action) for action in actions])

        return actions[arg_max]


    def get_action(self, state):
        """
          Returns: the action to take in the current state.  With probability self.epsilon,
                   take a random action and take the best policy action otherwise.  If there are
                   no legal actions, which is the case at the terminal state, returns None.
        """

        # Pick Action
        legal_actions = state.get_legal_actions()
        action = None

        if not legal_actions:
            return None

        if flip_coin(self.epsilon):
            action_idx = np.random.choice(len(legal_actions))
            action = legal_actions[action_idx]
        else:
            action = self.compute_action_from_q_values(state, legal_actions)

        self.do_action(state, action)
        return action


    def update(self, state, action, next_state, reward):

        features = np.array(self.feature_func(state, action)).reshape(-1,1)

        expected = reward + self.gamma * self.compute_value_from_q_values(next_state)
        current = self.get_q_value(state, action)
        
#         self.nn.fit(features, np.array([expected]), self.params)

        temporal_difference = expected - current
        this_grad = self.grad_q_value(features)(self.nn.weights).reshape(1,-1)
        this_grad[~np.isfinite(this_grad)] = 0.

        self.nn.weights = self.nn.weights + self.alpha * temporal_difference * this_grad


        # self.mu = self.mu + self.beta * ((temporal_difference) * features - self.mu)
        # self.nn.weights = self.nn.weights + self.alpha * (features - self.gamma * )


    def getPolicy(self, state):
        return self.compute_action_from_q_values(state, state.get_legal_actions())


    def getValue(self, state):
        return self.compute_value_from_q_values(state)  


    def observe_transition(self, state, action, next_state, reward, next_action=None):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, reward)


    def observation_function(self, state):
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            # print('reward is', reward)
            self.observe_transition(self.prev_state, self.prev_action, state, reward)

    def update_parameters(self, freq, num_games):
        if num_games % freq == 0:
            self.original_alpha /= 2.0
            self.original_epsilon /= 2.0
            

class SarsaLearningAgent_MLP(QLearningAgent_MLP):

    def __init__(self, architecture, feature_func=checkers_features_augmented, feature_count=34, reward_function=checkers_reward, alpha=0.01, gamma=0.1, epsilon=0.5, is_learning_agent=True, weights=None):
        
        QLearningAgent_MLP.__init__(self, architecture, feature_func, feature_count,reward_function,alpha, gamma, epsilon, is_learning_agent, weights)
    
    def start_episode(self):
        # Accumulate rewards while training for each episode and show total rewards 
        # at the end of each episode i.e. when stop episode
        self.prev_state = None
        self.prev_action = None
        self.prev_prev_state = None
        self.prev_prev_action = None
        
        self.episode_rewards = 0.0

    def update(self, state, action, next_state, next_action, reward):

        features = np.array(self.feature_func(state, action)).reshape(-1,1)

        if next_action is None:
            next_q_value = 0.0
        else:
            next_q_value = \
            self.get_q_value(next_state, next_action)
    
        expected = reward + self.gamma * next_q_value

        current = self.get_q_value(state, action)

        temporal_difference = expected - current

        this_grad = self.grad_q_value(features)(self.nn.weights).reshape(1,-1)
        this_grad[~np.isfinite(this_grad)] = 0.

        self.nn.weights = self.nn.weights + self.alpha * temporal_difference * this_grad


    def observe_transition(self, state, action, next_state, next_action, reward):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, next_action, reward)
    
    def observation_function(self, state):
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            self.prev_prev_state = self.prev_state
            self.prev_prev_action = self.prev_action
            action = self.get_action(state)
            self.observe_transition(self.prev_prev_state, self.prev_prev_action, self.prev_state, self.prev_action, reward)

            return action


            
            
            
class MLP:
    """Implement a feed-forward neural network"""

    def __init__(self, architecture, random=None, weights=None):

        # Unpack model architecture
        self.params = {
            'dim_hidden': architecture['width'], # List of number of nodes per layer
            'dim_in': architecture['input_dim'],
            'dim_out': architecture['output_dim'],
            'activation_type': architecture['activation_fn_type'],
            'activation_params': architecture['activation_fn_params'],
        }
        self.activation = architecture['activation_fn']

        # Number of parameters (weights + biases) in input layer
        self.D_in = self.params['dim_in'] * self.params['dim_hidden'][0] + self.params['dim_hidden'][0]

        # Number of parameters (weights + biases) in hidden layers
        self.D_hidden = 0
        for i, h in enumerate(self.params['dim_hidden'][1:]):
            # Multiply previous layer width by current layer width plus the bias (current layer width)
            self.D_hidden += self.params['dim_hidden'][i] * h + h

        # Number of parameters (weights + biases) in output layer
        self.D_out = self.params['dim_hidden'][-1] * self.params['dim_out'] + self.params['dim_out']

        # Number of total parameters
        self.D = self.D_in + self.D_hidden + self.D_out

        # Set random state for reproducibility
        if random is not None:
            self.random = random
        else:
            self.random = np.random.RandomState(0)

        # Initiate model parameters (weights + biases)
        if weights is None:
            self.weights = self.random.normal(0, 0.01, size=(1, self.D))
        else:
            assert weights.shape == (1, self.D)
            self.weights = weights

        # To inspect model training later
        self.objective_trace = np.empty((1, 1))
        self.weight_trace = np.empty((1, self.D))


    def forward(self, weights, x, return_features=False):
        """Forward pass given weights and input data.

        Returns:
            output (numpy.array): Model predictions of shape (n_mod, n_param, n_obs).
        """
        ''' Forward pass given weights and input '''
        H = self.params['dim_hidden']
        dim_in = self.params['dim_in']
        dim_out = self.params['dim_out']

        assert weights.shape[1] == self.D

        if len(x.shape) == 2:
            assert x.shape[0] == dim_in
            x = x.reshape((1, dim_in, -1))
        else:
            assert x.shape[1] == dim_in

        weights = weights.T # Shape: (n_obs, n_param)

        #input to first hidden layer
        W = weights[:H[0] * dim_in].T.reshape((-1, H[0], dim_in))
        b = weights[H[0] * dim_in:H[0] * dim_in + H[0]].T.reshape((-1, H[0], 1))
        input = self.activation(np.matmul(W, x) + b)
        index = H[0] * dim_in + H[0]

        assert input.shape[1] == H[0]

        #additional hidden layers
        for i in range(len(self.params['dim_hidden']) - 1):
            W = weights[index:index + H[i] * H[i+1]].T.reshape((-1, H[i+1], H[i]))
            index += H[i] * H[i+1]
            b = weights[index:index + H[i+1]].T.reshape((-1, H[i+1], 1))
            index += H[i+1]
            output = np.matmul(W, input) + b
            input = self.activation(output)

            assert input.shape[1] == H[i+1]

        # Return values from the last hidden layer if desired
        if return_features:
            return input # Shape: (n_mod, n_param, n_obs)

        #output layer
        W = weights[index:index + H[-1] * dim_out].T.reshape((-1, dim_out, H[-1]))
        b = weights[index + H[-1] * dim_out:].T.reshape((-1, dim_out, 1))
        output = np.matmul(W, input) + b
        assert output.shape[1] == self.params['dim_out']

        return output


    def _make_objective(self, x_train, y_train, reg_param=0):
        ''' Make objective functions: depending on whether or not you want to apply l2 regularization '''

        def objective(W, t):
            squared_error = np.linalg.norm(y_train - self.forward(W, x_train), axis=1)**2
#             mean_error = np.mean(squared_error) + reg_param * np.linalg.norm(W) / W.size**0.5
            mean_error = np.mean(squared_error) + reg_param * (np.linalg.norm(W)**2) / W.size
            return mean_error

        return objective, grad(objective)


    def _fit(self, objective, gradient, params):

        self.objective, self.gradient = (objective, gradient)

        ### set up optimization
        step_size = params.get('step_size', 0.01)
        max_iteration = params.get('max_iteration', 5000)
        check_point = params.get('check_point', 100)
        weights_init = params.get('init', self.weights.reshape((1, -1)))
        mass = params.get('mass', None)
        optimizer = params.get('optimizer', 'adam')
        random_restarts = params.get('random_restarts', 5)

        # Define callback function
        def call_back(weights, iteration, g):
            ''' Actions per optimization step '''
            objective = self.objective(weights, iteration)
            self.objective_trace = np.vstack((self.objective_trace, objective))
            self.weight_trace = np.vstack((self.weight_trace, weights))
            if iteration % check_point == 0:
                print("Iteration {} loss {}; gradient mag: {}".format(iteration, objective, np.linalg.norm(self.gradient(weights, iteration))))
        call_back = params.get('call_back', call_back)

        ### train with random restarts
        optimal_obj = 1e16
        optimal_weights = self.weights

        for i in range(random_restarts):
            if optimizer == 'adam':
                adam(self.gradient, weights_init, step_size=step_size, num_iters=max_iteration, callback=call_back)
            local_opt = np.min(self.objective_trace[-100:])

            if local_opt < optimal_obj:
                opt_index = np.argmin(self.objective_trace[-100:])
                self.weights = self.weight_trace[-100:][opt_index].reshape((1, -1))
            weights_init = self.random.normal(0, 1, size=(1, self.D))

        self.objective_trace = self.objective_trace[1:]
        self.weight_trace = self.weight_trace[1:]


    def fit(self, x_train, y_train, params):
        ''' Wrapper for MLE through gradient descent '''
        assert x_train.shape[0] == self.params['dim_in']
        assert y_train.shape[0] == self.params['dim_out']

        # Make objective function for training
        reg_param = params.get('reg_param', 0) # No regularization by default
        objective, gradient = self._make_objective(x_train, y_train, reg_param)

        # Train model
        self._fit(objective, gradient, params)


    