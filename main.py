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
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.q_table = {s: {a: 0.0 for a in env.actions[s]} for s in env.states}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

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

if __name__ == "__main__":
    num_epochs = 500
    steps_per_epoch = 1000

    env = RecyclingRobotEnvironment(alpha=0.6, beta=0.8, r_search=10, r_wait=2)
    agent = TDAgent(env)
    
    rewards_per_epoch = []

    for epoch in range(num_epochs):
        current_state = env.reset()
        total_reward = 0
        
        for step in range(steps_per_epoch):
            action = agent.choose_action(current_state)
            next_state, reward = env.step(action)
            agent.update_q_table(current_state, action, reward, next_state)
            
            total_reward += reward
            current_state = next_state
            
        rewards_per_epoch.append(total_reward)
        print(f"Epoch {epoch+1}/{num_epochs}: Total Reward = {total_reward}")

    with open('rewards.txt', 'w') as f:
        for reward in rewards_per_epoch:
            f.write(f"{reward}\n")

    print("\nTraining complete. Rewards saved in rewards.txt.")
    print("Agent's final Q table:")
    for state, actions in agent.q_table.items():
            print(f"State: {state}")
            for action, q_value in actions.items():
                print(f" - Action '{action}': {q_value:.2f}")