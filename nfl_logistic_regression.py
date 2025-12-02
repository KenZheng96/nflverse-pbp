import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("LOGISTIC REGRESSION - NFL PLAY SUCCESS PREDICTION")
print("="*60)


print("\n[1/6] Loading data...")
df = pd.read_parquet("play_by_play_2023.parquet")

offensive_plays = df[
    (df['play_type'].isin(['pass', 'run'])) & 
    (df['season_type'] == 'REG') &
    (df['epa'].notna())
].copy()

print(f"Loaded {len(offensive_plays):,} offensive plays")




# Select features for modeling
features_to_use = ['down', 'ydstogo', 'score_differential', 'qtr', 
                   'yardline_100', 'play_type', 'shotgun', 'no_huddle']

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
print("\n[3/6] Splitting train/test data...")

X = model_data[['down', 'ydstogo', 'score_differential', 'qtr', 
                'yardline_100', 'play_type_encoded', 'shotgun', 'no_huddle']]
y = model_data['success']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} plays")
print(f"Test set: {len(X_test):,} plays")

# ============================================
# TRAIN LOGISTIC REGRESSION MODEL
# ============================================
print("\n[4/6] Training logistic regression model...")

log_reg = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    class_weight='balanced'
    )

log_reg.fit(X_train, y_train)

print("Model trained successfully")

# ============================================
# EVALUATE MODEL
# ============================================
print("\n[5/6] Evaluating model performance...")

# Predictions
y_pred_train = log_reg.predict(X_train)
y_pred_test = log_reg.predict(X_test)

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

# ============================================
# FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================
print("\n[6/6] Analyzing feature importance...")

# Get coefficients
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_reg.coef_[0],
    'abs_coefficient': np.abs(log_reg.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (LOGISTIC REGRESSION COEFFICIENTS)")
print(f"{'='*60}")
print("\nInterpretation: Positive = increases success odds")
print("                Negative = decreases success odds")
print(f"{'='*60}\n")
print(feature_importance[['feature', 'coefficient']].to_string(index=False))
print(f"{'='*60}")

# ============================================
# VISUALIZATIONS
# ============================================
print("\n[Bonus] Creating visualizations...")

# 1. Feature Importance Chart
plt.figure(figsize=(10, 6))
feature_plot = feature_importance.sort_values('coefficient')
colors = ['red' if x < 0 else 'green' for x in feature_plot['coefficient']]
plt.barh(feature_plot['feature'], feature_plot['coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Logistic Regression Feature Importance\n(How each factor affects success odds)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')
print("Saved: logistic_regression_coefficients.png")

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Fail', 'Predicted Success'],
            yticklabels=['Actual Fail', 'Actual Success'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('logistic_regression_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: logistic_regression_confusion_matrix.png")

# 3. Prediction Probability Distribution
plt.figure(figsize=(10, 6))
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]
plt.hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
         bins=30, label=['Failed Plays', 'Successful Plays'], 
         color=['red', 'green'], alpha=0.7)
plt.xlabel('Predicted Probability of Success')
plt.ylabel('Number of Plays')
plt.title('Distribution of Success Probabilities')
plt.legend()
plt.tight_layout()
plt.savefig('logistic_regression_probability_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: logistic_regression_probability_distribution.png")

# ============================================
# KEY INSIGHTS
# ============================================
print(f"\n{'='*60}")
print("KEY INSIGHTS FROM LOGISTIC REGRESSION")
print(f"{'='*60}")

# Most important features
top_3_positive = feature_importance.nlargest(3, 'coefficient')
top_3_negative = feature_importance.nsmallest(3, 'coefficient')

print("TOP FACTORS THAT INCREASE SUCCESS:")
for idx, row in top_3_positive.iterrows():
    print(f"   {row['feature']:20s}: +{row['coefficient']:.4f}")

print("TOP FACTORS THAT DECREASE SUCCESS:")
for idx, row in top_3_negative.iterrows():
    print(f"   {row['feature']:20s}: {row['coefficient']:.4f}")

print(f"\n{'='*60}")
print(f"Model can predict play success with {test_accuracy*100:.1f}% accuracy")
print(f"This means {100-test_accuracy*100:.1f}% of plays defy simple patterns")
print(f"{'='*60}\n")

print("Analysis complete! Check your directory for PNG charts.")