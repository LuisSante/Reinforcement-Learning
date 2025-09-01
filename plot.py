import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from robot import RecyclingRobotEnvironment, TDAgent
import numpy as np

# Improved analytics-based configuration
num_epochs = 500
steps_per_epoch = 1000

# Optimized environment parameters
env = RecyclingRobotEnvironment(alpha=0.6, beta=0.8, r_search=10, r_wait=2)

# Agent with improved parameters for better learning
agent = TDAgent(
    env, 
    learning_rate=0.15,      # Slightly higher for faster initial learning
    discount_factor=0.95,    # Greater importance to future rewards
    epsilon=0.3,             # Higher initial exploration
    epsilon_decay=0.998,     # Gradual decay of exploration
    epsilon_min=0.01,        # Minimum exploration to maintain adaptability
    lr_decay=0.9995,         # Very gradual decay of learning rate
    lr_min=0.01              # Minimum learning rate
)

# Training with progress tracking
rewards_per_epoch = []
cumulative_rewards = []
total_cumulative = 0

print("Starting enhanced training")

for epoch in range(num_epochs):
    current_state = env.reset()
    total_reward = 0
    
    for step in range(steps_per_epoch):
        action = agent.choose_action(current_state)
        next_state, reward = env.step(action)
        agent.update_q_table(current_state, action, reward, next_state)
        
        total_reward += reward
        current_state = next_state
    
    total_cumulative += total_reward
    rewards_per_epoch.append(total_reward)
    cumulative_rewards.append(total_cumulative)
    
    # Show progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        params = agent.get_current_params()
        avg_last_100 = np.mean(rewards_per_epoch[-100:])
        print(f"Epoch {epoch+1:3d}: Reward = {total_reward:6.0f} | "
              f"Avg(100) = {avg_last_100:6.0f} | "
              f"ε = {params['epsilon']:.4f} | "
              f"α = {params['learning_rate']:.4f}")

print("Training completed!")

# Save rewards as in the original
with open('rewards.txt', 'w') as f:
    for reward in cumulative_rewards:  # Save cumulative rewards for consistency
        f.write(f"{reward}\n")

# Q-table analysis as in the original
search = []
wait = []
reload = []

print(f"\nQ-Table Final:")
print("=" * 40)
for state, actions in agent.q_table.items():
    print(f"State: {state}")
    for action, q_value in actions.items():
        print(f"  {action}: {q_value:.4f}")
        if action == 'search':
            search.append(q_value)
        elif action == 'wait':
            wait.append(q_value)
        elif action == 'reload':
            reload.append(q_value)

print(f"\nQ-values summary:")
print(f"Search Q-values: {search}")
print(f"Wait Q-values: {wait}")
print(f"Reload Q-values: {reload}")

# Generar los mismos gráficos que el original
q_df = pd.DataFrame(agent.q_table).T
q_df = q_df.reindex(columns=['search', 'wait', 'reload']) 

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Heatmap de la política óptima (igual que original)
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

# Gráfico de recompensas acumulativas (igual que original)
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

# Additional improvement statistics
print(f"\n" + "=" * 50)
print("IMPROVEMENT STATISTICS:")
print("=" * 50)

# Calculate improvement in last 100 epochs vs first 100
first_100_avg = np.mean(rewards_per_epoch[:100])
last_100_avg = np.mean(rewards_per_epoch[-100:])
improvement = last_100_avg - first_100_avg

print(f"Average reward first 100 epochs: {first_100_avg:.0f}")
print(f"Average reward last 100 epochs:  {last_100_avg:.0f}")
print(f"Absolute improvement: {improvement:.0f}")
print(f"Percentage improvement: {(improvement/first_100_avg)*100:.1f}%")
print(f"Final total reward: {rewards[-1]:,}")

# Convergence analysis
convergence_threshold = 0.05  # 5% variation
recent_rewards = rewards_per_epoch[-50:]
std_recent = np.std(recent_rewards)
mean_recent = np.mean(recent_rewards)
cv = std_recent / mean_recent if mean_recent != 0 else float('inf')

if cv < convergence_threshold:
    print(f"✓ Algorithm CONVERGED (CV = {cv:.4f} < {convergence_threshold})")
else:
    print(f"⚠ Algorithm has NOT fully CONVERGED (CV = {cv:.4f})")

print(f"\nFinal parameters of the agent:")
final_params = agent.get_current_params()
print(f"Final epsilon: {final_params['epsilon']:.6f}")
print(f"Final learning rate: {final_params['learning_rate']:.6f}")
print(f"Total steps: {final_params['step_count']:,}")