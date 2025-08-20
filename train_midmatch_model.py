# train_midmatch_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("IPL.csv", low_memory=False)

# Mid-match features and target
feature_cols = [
    "batting_team",
    "bowling_team",
    "venue",
    "city",
    "team_runs",
    "team_wicket",
    "overs",
]
target_col = "match_won_by"

# Keep only rows with required data
df = df[feature_cols + [target_col]].dropna()

# Clean string columns
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype(str).str.strip()

# Ensure numeric columns are numeric
df["team_runs"] = pd.to_numeric(df["team_runs"], errors="coerce")
df["team_wicket"] = pd.to_numeric(df["team_wicket"], errors="coerce")
df["overs"] = pd.to_numeric(df["overs"], errors="coerce")
df = df.dropna(subset=["team_runs", "team_wicket", "overs", target_col])

# Fit LabelEncoders for categorical columns (and target)
encoders = {}
categorical_cols = ["batting_team", "bowling_team", "venue", "city", target_col]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Prepare X and y
X = df[feature_cols]
y = df[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model + encoders + metadata
with open("midmatch_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "encoders": encoders,
            "feature_cols": feature_cols,
            "target_col": target_col,
        },
        f,
    )

print("Saved midmatch_model.pkl (model + encoders).")
