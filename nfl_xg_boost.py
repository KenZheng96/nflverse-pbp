import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("XGBOOST - NFL PLAY SUCCESS PREDICTION")
print("="*60)

# ============================================
# LOAD DATA
# ============================================
print("\n[1/7] Loading data...")
df = pd.read_parquet("play_by_play_2023.parquet")

offensive_plays = df[
    (df['play_type'].isin(['pass', 'run'])) & 
    (df['season_type'] == 'REG') &
    (df['epa'].notna())
].copy()

print(f"Loaded {len(offensive_plays):,} offensive plays")

# ============================================
# PREPARE MODELING DATA
# ============================================
print("\n[2/7] Preparing features...")

# Select features for modeling (more than logistic regression can handle well)
features_to_use = ['down', 'ydstogo', 'score_differential', 'qtr', 
                   'yardline_100', 'play_type', 'shotgun', 'no_huddle',
                   'wp', 'half_seconds_remaining', 'posteam_timeouts_remaining']

model_data = offensive_plays[features_to_use].copy()
model_data['success'] = (offensive_plays['epa'] > 0).astype(int)

# Drop rows with missing values
model_data = model_data.dropna()

# Encode categorical variables
le = LabelEncoder()
model_data['play_type_encoded'] = le.fit_transform(model_data['play_type'])

print(f"Features prepared: {features_to_use}")
print(f"Dataset size: {len(model_data):,} plays")
print(f"Success rate: {model_data['success'].mean()*100:.1f}%")

# ============================================
# SPLIT DATA
# ============================================
print("\n[3/7] Splitting train/test data...")

X = model_data[['down', 'ydstogo', 'score_differential', 'qtr', 
                'yardline_100', 'play_type_encoded', 'shotgun', 'no_huddle',
                'wp', 'half_seconds_remaining', 'posteam_timeouts_remaining']]
y = model_data['success']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} plays")
print(f"Test set: {len(X_test):,} plays")

# ============================================
# TRAIN XGBOOST MODEL
# ============================================
print("\n[4/7] Training XGBoost model...")
print("   (This may take 30-60 seconds...)")

xgb_model = xgb.XGBClassifier(
    n_estimators=50,          # Fewer trees (was 100)
    max_depth=4,              # Shallower trees (was 6)
    learning_rate=0.1,
    min_child_weight=5,       # NEW: Require more samples per leaf
    gamma=0.1,                # NEW: Minimum loss reduction for split
    subsample=0.8,            # NEW: Use 80% of data per tree
    colsample_bytree=0.8,     # NEW: Use 80% of features per tree
    reg_alpha=0.1,            # NEW: L1 regularization
    reg_lambda=1.0,           # NEW: L2 regularization
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

print("Model trained successfully")

# ============================================
# EVALUATE MODEL
# ============================================
print("\n[5/7] Evaluating model performance...")

# Predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n{'='*60}")
print("MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Training Accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
print(f"Test Accuracy:     {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
print(f"{'='*60}")

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]:,}")
print(f"False Positives: {cm[0,1]:,}")
print(f"False Negatives: {cm[1,0]:,}")
print(f"True Positives:  {cm[1,1]:,}")

# Detailed Classification Report
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred_test, 
                          target_names=['Failed Play', 'Successful Play']))

# ============================================
# FEATURE IMPORTANCE
# ============================================
print("\n[6/7] Analyzing feature importance...")

# Get feature importance from XGBoost
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (XGBOOST)")
print(f"{'='*60}")
print("\nInterpretation: Higher = more important for predictions")
print("                (Sum of all importances = 1.0)")
print(f"{'='*60}\n")
print(feature_importance.to_string(index=False))
print(f"{'='*60}")

# ============================================
# VISUALIZATIONS
# ============================================
print("\n[7/7] Creating visualizations...")

# 1. Feature Importance Chart
plt.figure(figsize=(10, 7))
feature_plot = feature_importance.sort_values('importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_plot)))
plt.barh(feature_plot['feature'], feature_plot['importance'], color=colors)
plt.xlabel('Influence')
plt.title('XGBoost Feature Importance\n(Which factors matter most for predicting success?)')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: xgboost_feature_importance.png")

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Predicted Fail', 'Predicted Success'],
            yticklabels=['Actual Fail', 'Actual Success'])
plt.title('Confusion Matrix - XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: xgboost_confusion_matrix.png")

# 3. Feature Importance Pie Chart (Top 6 features)
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(6)
other_importance = feature_importance.iloc[6:]['importance'].sum()

plot_data = pd.concat([
    top_features,
    pd.DataFrame({'feature': ['Other'], 'importance': [other_importance]})
])

colors_pie = plt.cm.Set3(range(len(plot_data)))
plt.pie(plot_data['importance'], labels=plot_data['feature'], autopct='%1.1f%%',
        colors=colors_pie, startangle=90)
plt.title('Feature Importance Distribution\n(What drives play success?)')
plt.tight_layout()
plt.savefig('xgboost_importance_pie.png', dpi=300, bbox_inches='tight')
print("Saved: xgboost_importance_pie.png")

# 4. Prediction Probability Distribution
plt.figure(figsize=(10, 6))
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
plt.hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
         bins=30, label=['Failed Plays', 'Successful Plays'], 
         color=['#d62728', '#2ca02c'], alpha=0.7)
plt.xlabel('Predicted Probability of Success')
plt.ylabel('Number of Plays')
plt.title('XGBoost: Distribution of Success Probabilities')
plt.legend()
plt.tight_layout()
plt.savefig('xgboost_probability_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: xgboost_probability_distribution.png")

# 5. Model Comparison Bar Chart (if you have logistic regression results)
plt.figure(figsize=(8, 6))
models = ['XGBoost', 'Baseline\n(50% guess)']
accuracies = [test_accuracy, 0.5]
colors_bar = ['#2ca02c', '#d62728']
plt.bar(models, accuracies, color=colors_bar, alpha=0.8)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim([0, 1])
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('xgboost_model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: xgboost_model_comparison.png")

# ============================================
# KEY INSIGHTS
# ============================================
print(f"\n{'='*60}")
print("KEY INSIGHTS FROM XGBOOST")
print(f"{'='*60}")

# Top 5 most important features
top_5 = feature_importance.head(5)

print("\nTOP 5 MOST IMPORTANT FACTORS FOR PLAY SUCCESS:")
for idx, row in top_5.iterrows():
    print(f"   {idx+1}. {row['feature']:30s}: {row['importance']:.3f} ({row['importance']*100:.1f}%)")

total_top5_importance = top_5['importance'].sum()
print(f"These 5 factors account for {total_top5_importance*100:.1f}% of prediction power")

print(f"\n{'='*60}")
print(f"XGBoost Accuracy: {test_accuracy*100:.1f}%")
print(f"Improvement over random guess: +{(test_accuracy-0.5)*100:.1f} percentage points")
print(f"{'='*60}\n")

# Answer the research question directly
print("="*60)
print("ANSWERING: 'What factors contribute to positive plays?'")
print("="*60)
most_important = feature_importance.iloc[0]
print(f"\nMost important factor: {most_important['feature']}")
print(f"Importance: {most_important['importance']*100:.1f}%")
print(f"\nThis means {most_important['feature']} has the strongest")
print("predictive power for whether a play succeeds or fails.")
print("="*60 + "\n")

print("XGBoost analysis complete! Check your directory for PNG charts.")