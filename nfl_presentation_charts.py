import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("GENERATING PRESENTATION CHARTS")
print("="*60)

# Set style for all charts
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 12

# ============================================
# CHART 1: SITUATION VS STRATEGY COMPARISON
# ============================================
print("\n[1/6] Creating Situation vs Strategy chart...")

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Situational Factors\n(Can\'t Control)', 'Strategic Choices\n(Can Control)']
importance = [0.415, 0.231]  # 41.5% and 23.1%
colors = ['#FF6B6B', '#4ECDC4']  # Red for uncontrollable, Teal for controllable

bars = ax.barh(categories, importance, color=colors, alpha=0.8, height=0.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, importance)):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val*100:.1f}%', 
            ha='left', va='center', fontsize=16, fontweight='bold')


ax.set_xlabel('Combined Importance', fontsize=14, fontweight='bold')
ax.set_title('Situation vs Strategy: What Matters More?', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0, 0.5)

# Format x-axis as percentages
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%'])

plt.tight_layout()
plt.savefig('situation_vs_strategy.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: situation_vs_strategy.png")
plt.close()


# ============================================
# CHART 2: 3RD DOWN CONVERSION RATES
# ============================================
print("\n[2/6] Creating 3rd down conversion chart...")

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Short\n(1-3 yards)', 'Medium\n(4-7 yards)', 'Long\n(8+ yards)']
conversion_rates = [0.59, 0.43, 0.23]
colors_gradient = ['#2ECC71', '#F39C12', '#E74C3C']  # Green to red

bars = ax.barh(categories, conversion_rates, color=colors_gradient, alpha=0.85, height=0.6)

# Add percentage labels
for bar, val in zip(bars, conversion_rates):
    width = bar.get_width()
    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val*100:.0f}%', 
            ha='left', va='center', fontsize=16, fontweight='bold')

ax.set_xlabel('Conversion Rate', fontsize=14, fontweight='bold')
ax.set_title('3rd Down Success Rate by Distance to Go', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0, 0.7)

# Format x-axis
ax.set_xticks([0, 0.2, 0.4, 0.6])
ax.set_xticklabels(['0%', '20%', '40%', '60%'])


plt.tight_layout()
plt.savefig('third_down_conversions.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: third_down_conversions.png")
plt.close()


# ============================================
# CHART 3: RED ZONE RUN VS PASS
# ============================================
print("\n[3/6] Creating red zone comparison chart...")

fig, ax = plt.subplots(figsize=(10, 6))

play_types = ['Run Plays', 'Pass Plays']
epa_values = [0.035, -0.071]
colors_rz = ['#27AE60', '#E74C3C']  # Green for positive, red for negative

bars = ax.bar(play_types, epa_values, color=colors_rz, alpha=0.85, width=0.5)

# Add value labels
for bar, val in zip(bars, epa_values):
    height = bar.get_height()
    label_y = height + 0.005 if height > 0 else height - 0.01
    ax.text(bar.get_x() + bar.get_width()/2, label_y,
            f'{val:+.3f} EPA',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=16, fontweight='bold')

# Add zero line
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

ax.set_ylabel('Expected Points Added (EPA)', fontsize=14, fontweight='bold')
ax.set_title('Red Zone Efficiency: Run vs Pass\n(Inside 20-Yard Line)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_ylim(-0.1, 0.06)



plt.tight_layout()
plt.savefig('redzone_run_vs_pass.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: redzone_run_vs_pass.png")
plt.close()


# ============================================
# CHART 4: FIELD POSITION EPA
# ============================================
print("\n[4/6] Creating field position EPA chart...")

fig, ax = plt.subplots(figsize=(10, 6))

positions = ['Red Zone\n(0-20)', 'Opponent Territory\n(20-50)', 
             'Own Territory\n(50-80)', 'Deep Own Territory\n(80-100)']
epa_values = [-0.018, -0.027, -0.032, -0.071]

# Color gradient from green (best) to red (worst)
colors_fp = ['#52C41A', '#FAAD14', '#FA8C16', '#F5222D']

bars = ax.barh(positions, epa_values, color=colors_fp, alpha=0.85, height=0.6)

# Add value labels
for bar, val in zip(bars, epa_values):
    width = bar.get_width()
    ax.text(width - 0.003, bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', 
            ha='right', va='center', fontsize=14, fontweight='bold', color='white')

# Add vertical line at zero
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)

ax.set_xlabel('Average EPA', fontsize=14, fontweight='bold')
ax.set_title('EPA by Field Position\n(Lower is Worse)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(-0.08, 0.01)

# Add labels
ax.text(-0.075, 3.5, 'Best', fontsize=12, fontweight='bold', color='#52C41A')
ax.text(-0.075, -0.5, 'Worst', fontsize=12, fontweight='bold', color='#F5222D')

plt.tight_layout()
plt.savefig('field_position_epa.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: field_position_epa.png")
plt.close()


# ============================================
# CHART 5: MODEL ACCURACY COMPARISON (Enhanced)
# ============================================
print("\n[5/6] Creating enhanced model comparison chart...")

fig, ax = plt.subplots(figsize=(10, 6))

models = ['Random\nBaseline', 'Logistic\nRegression', 'XGBoost']
accuracies = [0.500, 0.583, 0.591]
colors_model = ['#95A5A6', '#3498DB', '#27AE60']

bars = ax.bar(models, accuracies, color=colors_model, alpha=0.85, width=0.5)

# Add value labels
for bar, val in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
            f'{val*100:.1f}%',
            ha='center', va='bottom', fontsize=16, fontweight='bold')



ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison\n(Test Set Accuracy)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_ylim(0.4, 0.75)

# Format y-axis as percentages
ax.set_yticks([0.4, 0.5, 0.6, 0.7])
ax.set_yticklabels(['40%', '50%', '60%', '70%'])

# Add baseline reference line
ax.axhline(y=0.5, color='#95A5A6', linestyle='--', linewidth=1.5, alpha=0.5, 
           label='Random Guess Baseline')

plt.tight_layout()
plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: model_accuracy_comparison.png")
plt.close()


# ============================================
# CHART 6: FEATURE IMPORTANCE WITH HIGHLIGHTS
# ============================================
print("\n[6/6] Creating enhanced feature importance chart...")

fig, ax = plt.subplots(figsize=(12, 7))

features = ['posteam_timeouts_remaining', 'score_differential', 'no_huddle', 
            'qtr', 'half_seconds_remaining', 'wp', 'yardline_100', 
            'shotgun', 'down', 'play_type_encoded', 'ydstogo']
importance = [0.038287, 0.043893, 0.047221, 0.050612, 0.054874, 
              0.055771, 0.062751, 0.070054, 0.152421, 0.160883, 0.263234]

# Create color array - highlight top 3
colors_imp = ['#95A5A6'] * 8 + ['#F39C12', '#E67E22', '#E74C3C']  # Gray for others, orange-red gradient for top 3

bars = ax.barh(features, importance, color=colors_imp, alpha=0.85)

# Add value labels
for bar, val in zip(bars, importance):
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{val*100:.1f}%', 
            ha='left', va='center', fontsize=11, fontweight='bold')

# Highlight the king
ax.patches[-1].set_edgecolor('#FFD700')  # Gold border for "ydstogo"
ax.patches[-1].set_linewidth(3)

# Add crown emoji or text for top feature
ax.text(0.27, 10.3, '👑 KING', fontsize=14, fontweight='bold', color='#FFD700')

ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
ax.set_title('Feature Importance Rankings\n(Which Factors Matter Most?)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0, 0.3)

# Format x-axis
ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
ax.set_xticklabels(['0%', '5%', '10%', '15%', '20%', '25%', '30%'])

# Add top 5 annotation
ax.axvline(x=0.07, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.3)
ax.text(0.15, 4, 'Top 5 = 70.9%\nof predictive power', 
        fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD', alpha=0.9))

plt.tight_layout()
plt.savefig('feature_importance_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: feature_importance_enhanced.png")
plt.close()


# ============================================
# BONUS: DATA PIPELINE FUNNEL
# ============================================
print("\n[BONUS] Creating data pipeline funnel chart...")

fig, ax = plt.subplots(figsize=(10, 6))

stages = ['Raw Data', 'After Filtering', 'Final Dataset']
play_counts = [50000, 33957, 33836]
colors_funnel = ['#E74C3C', '#F39C12', '#27AE60']

# Create funnel effect using horizontal bars of decreasing width
y_positions = [2, 1, 0]
bar_heights = [0.8, 0.6, 0.4]

for i, (stage, count, color, height) in enumerate(zip(stages, play_counts, colors_funnel, bar_heights)):
    ax.barh(y_positions[i], count, height=height, color=color, alpha=0.85)
    ax.text(count + 1000, y_positions[i], f'{count:,} plays', 
            va='center', fontsize=14, fontweight='bold')

ax.set_yticks(y_positions)
ax.set_yticklabels(stages, fontsize=14)
ax.set_xlabel('Number of Plays', fontsize=14, fontweight='bold')
ax.set_title('Data Preprocessing Pipeline\n(Funnel View)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlim(0, 55000)

# Add reduction annotations
ax.annotate(f'-{50000-33957:,} plays', xy=(42000, 1.5), 
            fontsize=11, style='italic', color='#E74C3C')
ax.annotate(f'-{33957-33836:,} plays', xy=(33900, 0.5), 
            fontsize=11, style='italic', color='#F39C12')

plt.tight_layout()
plt.savefig('data_pipeline_funnel.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: data_pipeline_funnel.png")
plt.close()


# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("CHART GENERATION COMPLETE!")
print("="*60)
print("\nGenerated Charts:")
print("1. situation_vs_strategy.png - Slide 6")
print("2. third_down_conversions.png - Slide 9")
print("3. redzone_run_vs_pass.png - Slide 9")
print("4. field_position_epa.png - Slide 9")
print("5. model_accuracy_comparison.png - Slide 7")
print("6. feature_importance_enhanced.png - Slide 4 (alternative)")
print("7. data_pipeline_funnel.png - Slide 3 (bonus)")
print("\nAll charts saved at 300 DPI for presentation quality!")
print("="*60)