import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_parquet("play_by_play_2023.parquet")
offensive_plays = df[
    (df['play_type'].isin(['pass', 'run'])) & 
    (df['season_type'] == 'REG') &
    (df['epa'].notna())
].copy()




offensive_plays['score_diff_group'] = pd.cut(
    offensive_plays['score_differential'],
    bins=[-100, -14, -7, 0, 7, 14, 100],
    labels=['Down 14+', 'Down 8-14', 'Down 1-7', 'Up 1-7', 'Up 8-14', 'Up 14+']
)

score_epa = offensive_plays.groupby('score_diff_group')['epa'].mean()

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('white')

# Color gradient: red (losing) to green (winning)
colors_score = ['#d62728', '#ff7f0e', '#ffbb78', '#90ee90', '#2ca02c', '#006400']
bars = ax.bar(range(len(score_epa)), score_epa.values, 
              color=colors_score, alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax.set_ylabel('Average EPA', fontsize=14, fontweight='bold')
ax.set_xlabel('Score Situation', fontsize=14, fontweight='bold')
ax.set_title('EPA by Score Differential', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(score_epa)))
ax.set_xticklabels(score_epa.index, rotation=0, ha='center', fontsize=11)
ax.set_ylim(-0.13, 0.02)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, v in enumerate(score_epa.values):
    y_pos = v + 0.005 if v > 0 else v - 0.008
    ax.text(i, y_pos, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('3_score_epa.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print("✓ Saved: 3_score_epa.png")
