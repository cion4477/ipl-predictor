# train_prematch_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset (make sure IPL.csv is in same folder)
df = pd.read_csv("IPL.csv", low_memory=False)

# Features & target (must match the app)
feature_cols = [
    "batting_team",
    "bowling_team",
    "toss_winner",
    "toss_decision",
    "venue",
    "city",
    "season",
    "match_type",
]
target_col = "match_won_by"

# Keep only rows that have all required fields
df = df[feature_cols + [target_col]].dropna()

# Strip strings to avoid whitespace mismatches
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

# Fit label encoders for categorical columns (including target)
encoders = {}
for col in feature_cols + [target_col]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Prepare X and y
X = df[feature_cols]
y = df[target_col]

# Train-test split (optional)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (RandomForest for stability)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model + encoders + metadata
with open("prematch_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "encoders": encoders,
            "feature_cols": feature_cols,
            "target_col": target_col,
        },
        f,
    )

print("Saved prematch_model.pkl (model + encoders).")
