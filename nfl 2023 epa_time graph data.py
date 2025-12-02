import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_parquet("play_by_play_2023.parquet")
offensive_plays = df[
    (df['play_type'].isin(['pass', 'run'])) & 
    (df['season_type'] == 'REG') &
    (df['epa'].notna())
].copy()





quarter_epa = offensive_plays.groupby('qtr')['epa'].mean()

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('white')

# Color progression through game
colors_time = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
bars = ax.bar(range(len(quarter_epa)), quarter_epa.values, 
              color=colors_time[:len(quarter_epa)], alpha=0.8, edgecolor='black', linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
ax.set_ylabel('Average EPA', fontsize=14, fontweight='bold')
ax.set_xlabel('Quarter', fontsize=14, fontweight='bold')
ax.set_title('EPA by Quarter', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(quarter_epa)))
ax.set_xticklabels([f'Q{int(q)}' if q <= 4 else 'OT' for q in quarter_epa.index], fontsize=12)
ax.set_ylim(-0.04, 0.01)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, v in enumerate(quarter_epa.values):
    y_pos = v + 0.002 if v > 0 else v - 0.004
    ax.text(i, y_pos, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)
    # Add play count
    count = len(offensive_plays[offensive_plays['qtr'] == quarter_epa.index[i]])
    ax.text(i, -0.037, f'{count:,} plays', ha='center', fontsize=9, style='italic')

# Add annotation
ax.text(1.5, 0.005, 'EPA relatively consistent across quarters', 
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.4))

plt.tight_layout()
plt.savefig('4_quarter_epa.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print("✓ Saved: 4_quarter_epa.png")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("ALL 4 GRAPHS CREATED SUCCESSFULLY!")
print("="*60)
print("\n1. 1_overall_epa.png - General pass vs run EPA")
print("2. 2_redzone_epa.png - Red zone pass vs run EPA (KEY FINDING)")
print("3. 3_score_epa.png - EPA by score differential")
print("4. 4_quarter_epa.png - EPA by quarter/time")
print("\n" + "="*60)