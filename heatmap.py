import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

q_data = {
    'search': [70.48, 59.49],
    'wait': [67.09, 60.15],
    'reload': [np.nan, 63.88] 
}

q_df = pd.DataFrame(q_data, index=['high', 'low'])

plt.figure(figsize=(8, 5))
sns.heatmap(q_df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
plt.title('Mapa de Calor de la Política Óptima')
plt.xlabel('Acciones')
plt.ylabel('Estados')
plt.show()