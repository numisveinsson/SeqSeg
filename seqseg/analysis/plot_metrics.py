import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# Load data
df = pd.read_csv("/Users/nsveinsson/Downloads/taubin_exp_smooth_mesh_stats.csv")

# Filter metrics
lap_df = df[df['metric'].str.contains('lap', case=False, na=False)]
norm_df = df[df['metric'].str.contains('norm', case=False, na=False)]

# Merge
merged = pd.merge(
    lap_df,
    norm_df,
    on=['dataset', 'category'],
    suffixes=('_lap', '_norm')
)

# Sort by Laplacian (higher-valued models)
merged = merged.sort_values(by='mean_lap', ascending=False)

# Labels
labels = merged['dataset'] + " - " + merged['category']

# Create figure
fig, ax1 = plt.subplots()

# Left axis (Laplacian)
ax1.plot(labels, merged['mean_lap'], marker='o')
ax1.set_xlabel("Model / Dataset")
ax1.set_ylabel("Laplacian")

# Right axis (Normal Consistency)
ax2 = ax1.twinx()
ax2.plot(labels, merged['mean_norm'], marker='o')
ax2.set_ylabel("Normal Consistency")

# Formatting
plt.xticks(rotation=45, ha='right')
plt.title("Laplacian vs Normal Consistency (Dual Axis)")
fig.tight_layout()

plt.show()