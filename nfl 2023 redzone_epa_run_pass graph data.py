import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_parquet("play_by_play_2023.parquet")
offensive_plays = df[
    (df['play_type'].isin(['pass', 'run'])) & 
    (df['season_type'] == 'REG') &
    (df['epa'].notna())
].copy()



red_zone = offensive_plays[offensive_plays['yardline_100'] <= 20].copy()
red_zone_epa = red_zone.groupby('play_type')['epa'].mean()

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('white')

colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(red_zone_epa.index, red_zone_epa.values, 
              color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax.set_ylabel('Average EPA', fontsize=14, fontweight='bold')
ax.set_xlabel('Play Type', fontsize=14, fontweight='bold')
ax.set_title('Red Zone EPA: Pass vs Run', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(-0.1, 0.06)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, (play_type, v) in enumerate(red_zone_epa.items()):
    y_pos = v + 0.004 if v > 0 else v - 0.008
    color = '#2ca02c' if v > 0 else '#d62728'
    ax.text(i, y_pos, f'{v:.3f}', ha='center', fontweight='bold', fontsize=13, color=color)
    # Add count below
    count = len(red_zone[red_zone['play_type'] == play_type])
    ax.text(i, -0.09, f'n = {count:,}', ha='center', fontsize=10, style='italic')

# Add annotation for surprise finding
ax.text(0.5, 0.045, '← Run is MORE efficient in red zone!', 
        ha='center', fontsize=11, color='#2ca02c', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('2_redzone_epa.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print("✓ Saved: 2_redzone_epa.png")
