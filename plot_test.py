import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv('/u/raavi/multi-turn-multi-hop/plots_new/turns_vs_hops_table.csv')

# Drop the 'Total' row if it exists
df = df[df['Hop/Turns'].str.lower() != 'total']

# Set 'Hop/Turns' as index
df.set_index('Hop/Turns', inplace=True)

# Transpose to get turns on x-axis
df = df.transpose()

# Plot setup
colors = plt.get_cmap('Set2').colors  # Soft pastel palette
ax = df.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)

# Customization with larger font sizes
plt.title('Hop Distribution Across Turns', fontsize=18)
plt.xlabel('Turn', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Hop Type', title_fontsize=14, fontsize=12)
plt.tight_layout()

plt.savefig('stacked_bar_chart.png')  # or plt.show()
