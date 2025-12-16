import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

print("🚦 ML Model Training - 4-Way Intersection Traffic Optimization\n")
print("=" * 70)

# ===========================
# 1. LOAD DATASET
# ===========================
print("📂 Loading dataset...")
try:
    df = pd.read_csv('traffic_dataset_4way.csv')
    print(f"✅ Loaded {len(df):,} records")
    print(f"   Columns: {list(df.columns)}\n")
except FileNotFoundError:
    print("❌ Error: traffic_dataset_4way.csv not found!")
    print("   Please run realistic_traffic_gen.py first.")
    exit(1)

# ===========================
# 2. FEATURE ENGINEERING
# ===========================
print("⚙️  Feature Engineering...")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['time_of_day'] = df['hour'] + df['minute'] / 60.0

# Rush hour indicator
df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] < 9)) | \
                      ((df['hour'] >= 17) & (df['hour'] < 19))
df['is_rush_hour'] = df['is_rush_hour'].astype(int)

# Time period categorization
def get_time_period(hour):
    if 7 <= hour < 9:
        return 'morning_rush'
    elif 9 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 14:
        return 'lunch'
    elif 14 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 19:
        return 'evening_rush'
    elif 19 <= hour < 23:
        return 'evening'
    else:
        return 'night'

df['time_period'] = df['hour'].apply(get_time_period)

# Traffic density
df['traffic_density'] = df['vehicle_count'] / 150  # Assuming 150m lane

# Speed-congestion ratio
df['speed_congestion_ratio'] = df['avg_speed'] / (df['vehicle_count'] + 1)

# Turn complexity factor (left turns = harder)
turn_complexity = {'Left': 3, 'Straight': 1, 'Right': 0.5}
df['turn_complexity'] = df['turn_intention'].map(turn_complexity)

# Total turn queue
df['total_turn_queue'] = df['left_turn_queue'] + df['straight_queue'] + df['right_turn_queue']

# Left turn ratio (indicator of difficult traffic)
df['left_turn_ratio'] = df['left_turn_queue'] / (df['total_turn_queue'] + 1)

# Encode categorical variables
le_origin = LabelEncoder()
le_destination = LabelEncoder()
le_turn = LabelEncoder()

df['origin_encoded'] = le_origin.fit_transform(df['origin'])
df['destination_encoded'] = le_destination.fit_transform(df['destination'])
df['turn_encoded'] = le_turn.fit_transform(df['turn_intention'])

print("✅ Features created:")
print("   • Time-based: hour, time_of_day, is_rush_hour, time_period")
print("   • Traffic: traffic_density, speed_congestion_ratio")
print("   • Turn-based: turn_complexity, left_turn_ratio, total_turn_queue")
print("   • Encoded: origin, destination, turn_intention\n")

# ===========================
# 3. DATA EXPLORATION
# ===========================
print("📊 Data Exploration:")
print(f"\n🚦 Congestion Level Distribution:")
congestion_dist = df['congestion_level'].value_counts()
print(congestion_dist)
for level, count in congestion_dist.items():
    print(f"   {level:8} → {count/len(df)*100:.1f}%")

print(f"\n🔄 Turn Intention Analysis:")
turn_dist = df['turn_intention'].value_counts()
print(turn_dist)

print(f"\n📈 Average metrics by congestion level:")
print(df.groupby('congestion_level')[['vehicle_count', 'avg_speed', 'emergency_count', 
                                       'left_turn_queue', 'total_turn_queue']].mean().round(2))

print(f"\n🛣️  Congestion by Origin Direction:")
print(df.groupby('origin')['congestion_level'].value_counts().unstack(fill_value=0))

# ===========================
# 4. ANOMALY DETECTION
# ===========================
print("\n🔍 Training Anomaly Detection Model...")

anomaly_features = ['vehicle_count', 'avg_speed', 'speed_mps', 'traffic_density',
                    'left_turn_queue', 'total_turn_queue']
X_anomaly = df[anomaly_features].fillna(0)

iso_forest = IsolationForest(
    contamination=0.015,
    random_state=42,
    n_estimators=100
)
df['ml_anomaly_score'] = iso_forest.fit_predict(X_anomaly)
df['ml_anomaly_score'] = (df['ml_anomaly_score'] == -1).astype(int)

print(f"✅ Anomaly detection complete")
print(f"   • ML detected: {df['ml_anomaly_score'].sum():,} ({df['ml_anomaly_score'].mean()*100:.2f}%)")
print(f"   • Actual anomalies: {df['anomaly_flag'].sum():,} ({df['anomaly_flag'].mean()*100:.2f}%)")
agreement = ((df['ml_anomaly_score'] == df['anomaly_flag']).sum() / len(df) * 100)
print(f"   • Agreement: {agreement:.1f}%\n")

# ===========================
# 5. PREPARE ML FEATURES
# ===========================
print("🎯 Preparing features for congestion prediction...")

feature_columns = [
    'vehicle_count',
    'avg_speed',
    'emergency_count',
    'time_of_day',
    'is_rush_hour',
    'lane_id',
    'left_turn_queue',
    'straight_queue',
    'right_turn_queue',
    'total_turn_queue',
    'traffic_density',
    'speed_congestion_ratio',
    'turn_complexity',
    'left_turn_ratio',
    'origin_encoded',
    'destination_encoded',
    'turn_encoded'
]

X = df[feature_columns].copy()
y = df['congestion_level'].copy()

# Encode target
le_congestion = LabelEncoder()
y_encoded = le_congestion.fit_transform(y)
label_mapping = dict(zip(le_congestion.classes_, le_congestion.transform(le_congestion.classes_)))
print(f"✅ Label encoding: {label_mapping}")
print(f"   Classes: {list(le_congestion.classes_)}")

# ===========================
# 6. TRAIN/TEST SPLIT
# ===========================
print("\n📊 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   • Training: {len(X_train):,} samples")
print(f"   • Testing: {len(X_test):,} samples\n")

# ===========================
# 7. TRAIN RANDOM FOREST
# ===========================
print("🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handle class imbalance
)

rf_model.fit(X_train, y_train)
print("✅ Model training complete!\n")

# ===========================
# 8. MODEL EVALUATION
# ===========================
print("📈 Model Performance Evaluation\n")
print("=" * 70)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Overall Accuracy: {accuracy*100:.2f}%\n")

# Classification report
print("📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_congestion.classes_, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("🔢 Confusion Matrix:")
print("             Predicted →")
header = "Actual ↓     " + "".join([f"{cl:>10}" for cl in le_congestion.classes_])
print(header)
print("-" * 50)

for i, actual_label in enumerate(le_congestion.classes_):
    if i < cm.shape[0]:
        row_values = "".join([f"{cm[i][j]:>10}" if j < cm.shape[1] else f"{'0':>10}" 
                              for j in range(len(le_congestion.classes_))])
        print(f"{actual_label:12} {row_values}")

# Feature Importance
print("\n🔍 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    bar_length = int(row['importance'] * 50)
    bar = '█' * bar_length
    print(f"   {row['feature']:25} {bar} {row['importance']:.4f}")

# ===========================
# 9. SAVE MODELS
# ===========================
print("\n💾 Saving trained models...")
joblib.dump(rf_model, 'traffic_model_4way.pkl')
joblib.dump(le_congestion, 'label_encoder_congestion.pkl')
joblib.dump(le_origin, 'label_encoder_origin.pkl')
joblib.dump(le_destination, 'label_encoder_destination.pkl')
joblib.dump(le_turn, 'label_encoder_turn.pkl')
joblib.dump(iso_forest, 'anomaly_detector_4way.pkl')

model_config = {
    'feature_columns': feature_columns,
    'label_mapping': {str(k): int(v) for k, v in label_mapping.items()},  # Convert to native types
    'classes': [str(c) for c in le_congestion.classes_],
    'model_type': 'RandomForestClassifier',
    'n_estimators': 150,
    'accuracy': float(accuracy),
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'turn_complexity_map': {str(k): float(v) for k, v in turn_complexity.items()},
    'origin_classes': [str(c) for c in le_origin.classes_],
    'destination_classes': [str(c) for c in le_destination.classes_],
    'turn_classes': [str(c) for c in le_turn.classes_]
}

with open('model_config_4way.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print("✅ Saved:")
print("   • traffic_model_4way.pkl")
print("   • label_encoder_*.pkl (4 encoders)")
print("   • anomaly_detector_4way.pkl")
print("   • model_config_4way.json\n")

# ===========================
# 10. GENERATE PREDICTIONS
# ===========================
print("🔮 Generating predictions for entire dataset...")

df['predicted_congestion'] = le_congestion.inverse_transform(rf_model.predict(X))
df['prediction_confidence'] = rf_model.predict_proba(X).max(axis=1)

# Enhanced priority score calculation
def calculate_priority_score(row):
    """
    Enhanced priority scoring for 4-way intersection
    Considers turns, emergencies, and congestion
    """
    base_priority = 0
    
    # Emergency vehicles (HIGHEST PRIORITY)
    if row['emergency_count'] > 0:
        base_priority += 2000 * row['emergency_count']
    
    # Congestion level
    congestion_weights = {'High': 150, 'Medium': 75, 'Low': 20}
    base_priority += congestion_weights.get(row['predicted_congestion'], 0)
    
    # Vehicle count factor
    base_priority += row['vehicle_count'] * 3
    
    # Turn complexity (left turns need more priority)
    if row['turn_intention'] == 'Left':
        base_priority += row['left_turn_queue'] * 5  # Left turns harder
    elif row['turn_intention'] == 'Straight':
        base_priority += row['straight_queue'] * 2
    else:  # Right turn
        base_priority += row['right_turn_queue'] * 1  # Easiest
    
    # Speed penalty (slower = more urgent)
    if row['avg_speed'] < 3:
        base_priority += 100
    elif row['avg_speed'] < 7:
        base_priority += 50
    
    # Anomaly detection
    if row['ml_anomaly_score'] == 1:
        base_priority += 300
    
    return base_priority

df['priority_score'] = df.apply(calculate_priority_score, axis=1)

# ===========================
# 11. SAVE PREDICTIONS
# ===========================
print("💾 Saving predictions...\n")

# Full predictions
prediction_columns = [
    'id', 'timestamp', 'lane_id', 'origin', 'destination', 'turn_intention',
    'vehicle_type', 'priority_level', 'vehicle_count', 'avg_speed', 
    'emergency_count', 'left_turn_queue', 'straight_queue', 'right_turn_queue',
    'congestion_level', 'predicted_congestion', 'prediction_confidence',
    'priority_score', 'ml_anomaly_score', 'anomaly_flag'
]

df[prediction_columns].to_csv('predictions_4way_full.csv', index=False)
print("✅ Saved: predictions_4way_full.csv")

# Aggregated by lane and timestamp
lane_summary = df.groupby(['timestamp', 'lane_id', 'origin']).agg({
    'vehicle_count': 'first',
    'avg_speed': 'first',
    'emergency_count': 'first',
    'left_turn_queue': 'first',
    'straight_queue': 'first',
    'right_turn_queue': 'first',
    'predicted_congestion': lambda x: x.mode()[0] if len(x) > 0 else 'Low',
    'priority_score': 'max',
    'ml_anomaly_score': 'max'
}).reset_index()

lane_summary.to_csv('predictions_4way_by_lane.csv', index=False)
print("✅ Saved: predictions_4way_by_lane.csv (for C integration)")

# JSON format (sample for testing)
json_predictions = []
for timestamp in df['timestamp'].unique()[:50]:  # First 50 timestamps
    timestamp_data = df[df['timestamp'] == timestamp]
    
    entry = {
        'timestamp': str(timestamp),
        'lanes': []
    }
    
    for lane_id in sorted(timestamp_data['lane_id'].unique()):
        lane_data = timestamp_data[timestamp_data['lane_id'] == lane_id]
        
        # Aggregate turn data
        turn_summary = lane_data.groupby('turn_intention').size().to_dict()
        
        entry['lanes'].append({
            'lane_id': int(lane_id),
            'origin': str(lane_data['origin'].iloc[0]),
            'vehicle_count': int(lane_data['vehicle_count'].iloc[0]),
            'avg_speed': float(lane_data['avg_speed'].iloc[0]),
            'emergency_count': int(lane_data['emergency_count'].iloc[0]),
            'left_turn_queue': int(lane_data['left_turn_queue'].iloc[0]),
            'straight_queue': int(lane_data['straight_queue'].iloc[0]),
            'right_turn_queue': int(lane_data['right_turn_queue'].iloc[0]),
            'turn_distribution': turn_summary,
            'predicted_congestion': str(lane_data['predicted_congestion'].mode()[0]),
            'priority_score': int(lane_data['priority_score'].max()),
            'has_anomaly': bool(lane_data['ml_anomaly_score'].max() == 1)
        })
    
    json_predictions.append(entry)

with open('predictions_4way_sample.json', 'w') as f:
    json.dump(json_predictions, f, indent=2)

print("✅ Saved: predictions_4way_sample.json (50 timestamps)\n")

# ===========================
# 12. DETAILED ANALYSIS
# ===========================
print("=" * 70)
print("📊 COMPREHENSIVE ANALYSIS")
print("=" * 70)

print(f"\n🎯 Prediction Accuracy by Congestion Level:")
for level in le_congestion.classes_:
    actual = df[df['congestion_level'] == level]
    if len(actual) > 0:
        correct = actual[actual['predicted_congestion'] == level]
        acc = len(correct) / len(actual) * 100
        print(f"   {level:8} → {acc:.2f}% ({len(correct):,}/{len(actual):,})")

print(f"\n🔄 Prediction Accuracy by Turn Intention:")
for turn in df['turn_intention'].unique():
    turn_data = df[df['turn_intention'] == turn]
    correct = (turn_data['congestion_level'] == turn_data['predicted_congestion']).sum()
    acc = correct / len(turn_data) * 100
    print(f"   {turn:10} → {acc:.2f}% ({correct:,}/{len(turn_data):,})")

print(f"\n🛣️  Prediction Accuracy by Origin:")
for origin in df['origin'].unique():
    origin_data = df[df['origin'] == origin]
    correct = (origin_data['congestion_level'] == origin_data['predicted_congestion']).sum()
    acc = correct / len(origin_data) * 100
    print(f"   {origin:8} → {acc:.2f}% ({correct:,}/{len(origin_data):,})")

print(f"\n🚨 High Priority Situations:")
high_priority = df[df['priority_score'] > 1000]
print(f"   • Total: {len(high_priority):,} instances ({len(high_priority)/len(df)*100:.1f}%)")
print(f"   • With emergencies: {len(high_priority[high_priority['emergency_count'] > 0]):,}")
print(f"   • High congestion: {len(high_priority[high_priority['predicted_congestion'] == 'High']):,}")
print(f"   • Left turn dominated: {len(high_priority[high_priority['left_turn_queue'] > 5]):,}")

print(f"\n⚠️  Anomaly Detection Performance:")
print(f"   • ML detected: {df['ml_anomaly_score'].sum():,}")
print(f"   • Actual anomalies: {df['anomaly_flag'].sum():,}")
print(f"   • Agreement: {agreement:.1f}%")

if df['anomaly_flag'].sum() > 0:
    print(f"\n   Anomaly types detected:")
    anomaly_types = df[df['anomaly_flag'] == 1]['anomaly_type'].value_counts()
    for atype, count in anomaly_types.items():
        print(f"      • {atype}: {count}")

print(f"\n🏆 Model Performance Summary:")
print(f"   • Overall Accuracy: {accuracy*100:.2f}%")
print(f"   • Training samples: {len(X_train):,}")
print(f"   • Testing samples: {len(X_test):,}")
print(f"   • Features used: {len(feature_columns)}")
print(f"   • Most important feature: {feature_importance.iloc[0]['feature']}")
print(f"   • Average prediction confidence: {df['prediction_confidence'].mean():.3f}")

print("\n" + "=" * 70)
print("✅ ML TRAINING COMPLETE!")
print("=" * 70)

