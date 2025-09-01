import numpy as np
import random

class RecyclingRobotEnvironment:
    def __init__(self, alpha=0.5, beta=0.8, r_search=10, r_wait=1):
        self.alpha = alpha
        self.beta = beta
        self.r_search = r_search
        self.r_wait = r_wait
        self.r_recharge = 0
        self.r_battery_dead = -100

        self.states = ['high', 'low']
        self.actions = {
        'high': ['search', 'wait'],
        'low': ['search', 'wait', 'reload']
        }

    def reset(self):
        self.current_state = 'high'
        return self.current_state

    def step(self, action):
        if self.current_state == 'high':
            if action == 'search':
                reward = self.r_search
                if random.random() < self.alpha:
                    self.current_state = 'high'
                else:
                    self.current_state = 'low'
            elif action == 'wait':
                reward = self.r_wait
                self.current_state = 'high' 
        
        elif self.current_state == 'low':
            if action == 'search':
                reward = self.r_search
                if random.random() < self.beta:
                    self.current_state = 'low'
                else:
                    reward = self.r_battery_dead 
                    self.current_state = 'high'
            elif action == 'wait':
                reward = self.r_wait
                self.current_state = 'low' 
            elif action == 'reload':
                reward = self.r_recharge
                self.current_state = 'high' 
        
        return self.current_state, reward

class TDAgent:
    def __init__(self, env, learning_rate=0.15, discount_factor=0.95, epsilon=0.3, 
                 epsilon_decay=0.998, epsilon_min=0.01, lr_decay=0.9995, lr_min=0.01):
        self.env = env
        self.q_table = {s: {a: 0.0 for a in env.actions[s]} for s in env.states}
        
        # Improved parameters based on problem analysis
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = epsilon
        self.epsilon = epsilon

        # Decay parameters for better convergence
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr_decay = lr_decay
        self.lr_min = lr_min

        # Step counter for progress tracking
        self.step_count = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions[state])
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q_value = old_q_value + self.learning_rate * (
            reward + self.discount_factor * max_next_q - old_q_value
        )
        self.q_table[state][action] = new_q_value

        # Increment counter and apply decays
        self.step_count += 1
        self._apply_decay()
    
    def _apply_decay(self):
        """Applies gradual decay of parameters for better convergence"""
        # Decay epsilon (less exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Decay learning rate (smaller steps over time)
        if self.learning_rate > self.lr_min:
            self.learning_rate = max(self.lr_min, self.learning_rate * self.lr_decay)
    
    def get_current_params(self):
        """Auxiliary method to monitor current parameters"""
        return {
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'step_count': self.step_count
        }