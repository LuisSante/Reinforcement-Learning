import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from robot import RecyclingRobotEnvironment, TDAgent


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
    #print(f"Epoch {epoch+1}/{num_epochs}: Total Reward = {total_reward}")
    
with open('rewards.txt', 'w') as f:
    for reward in rewards_per_epoch:
        f.write(f"{reward}\n")

search = []
wait = []
reload = []
for state, actions in agent.q_table.items():
    print(f"State: {state}")
    for action, q_value in actions.items():
        if action == 'search':
            search.append(q_value)
        elif action == 'wait':
            wait.append(q_value)
        elif action == 'reload':
            reload.append(q_value)

print(f"Search Q-values: {search}")
print(f"Wait Q-values: {wait}")
print(f"Reload Q-values: {reload}")

q_df = pd.DataFrame(agent.q_table).T
q_df = q_df.reindex(columns=['search', 'wait', 'reload']) 
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

sns.heatmap(
    q_df, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    linewidths=.5,
    mask=q_df.isna(),
    cbar_kws={"label": "Q-value"},
    ax=axes[0]
)
axes[0].set_title('Optimal Policy Heat Map')
axes[0].set_xlabel('Actions')
axes[0].set_ylabel('States')

rewards = []
with open('rewards.txt', 'r') as f:
    for line in f:
        rewards.append(int(line.strip()))

axes[1].plot(rewards)
axes[1].set_title('Total Cumulative Reward by Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Total Cumulative Reward')
axes[1].grid(True)

plt.tight_layout()
plt.show()