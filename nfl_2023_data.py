import pandas as pd
import numpy as np

# Load the data
df = pd.read_parquet("play_by_play_2023.parquet")

# offense plays only, regular season
offensive_plays = df[
    (df['play_type'].isin(['pass', 'run'])) & 
    (df['season_type'] == 'REG') &
    (df['epa'].notna())
].copy()

print(f"Total offensive plays analyzed: {len(offensive_plays):,}")
print(f"Total number of variables per play: {len(offensive_plays.columns)}")


# FIELD POSITION ANALYSIS
print("\n=== EPA by Field Position ===")
offensive_plays['field_position'] = pd.cut(
    offensive_plays['yardline_100'],
    bins=[0, 20, 50, 80, 100],
    labels=['Red Zone (0-20)', 'Opponent Territory (20-50)', 
            'Own Territory (50-80)', 'Deep Own Territory (80-100)']
)
field_pos_analysis = offensive_plays.groupby('field_position').agg({
    'epa': 'mean',
    'success': 'mean',
    'play_id': 'count'
}).rename(columns={'play_id': 'total_plays'})
field_pos_analysis['success'] = field_pos_analysis['success'] * 100
print(field_pos_analysis)


# PLAY ACTION EFFECT
print("\n=== Play Action vs. Standard Pass ===")
pass_plays = offensive_plays[offensive_plays['play_type'] == 'pass'].copy()
play_action_analysis = pass_plays.groupby('pass_attempt').agg({
    'epa': 'mean',
    'success': 'mean',
    'play_id': 'count'
}).rename(columns={'play_id': 'total_plays'})
play_action_analysis['success'] = play_action_analysis['success'] * 100
print(play_action_analysis)


# 3RD DOWN CONVERSION ANALYSIS
print("\n=== 3rd Down Success by Distance ===")
third_downs = offensive_plays[offensive_plays['down'] == 3].copy()
third_downs['distance_category'] = pd.cut(
    third_downs['ydstogo'],
    bins=[0, 3, 7, 100],
    labels=['Short (1-3)', 'Medium (4-7)', 'Long (8+)']
)
third_down_success = third_downs.groupby('distance_category').agg({
    'epa': 'mean',
    'success': 'mean',
    'first_down': 'mean',
    'play_id': 'count'
}).rename(columns={'play_id': 'total_plays', 'first_down': 'conversion_rate'})
third_down_success[['success', 'conversion_rate']] *= 100
print(third_down_success)


# RED ZONE EFFICIENCY
print("\n=== Red Zone Performance ===")
red_zone = offensive_plays[offensive_plays['yardline_100'] <= 20].copy()
red_zone_stats = red_zone.groupby('play_type').agg({
    'epa': 'mean',
    'success': 'mean',
    'touchdown': 'mean',
    'play_id': 'count'
}).rename(columns={'play_id': 'total_plays', 'touchdown': 'td_rate'})
red_zone_stats[['success', 'td_rate']] *= 100
print(red_zone_stats)


