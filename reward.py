import matplotlib.pyplot as plt

rewards = []
with open('rewards.txt', 'r') as f:
    for line in f:
        rewards.append(int(line.strip()))

plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('Recompensa Total Acumulada por Época')
plt.xlabel('Época')
plt.ylabel('Recompensa Total Acumulada')
plt.grid(True)
plt.show()