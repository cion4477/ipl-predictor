from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)
BASE = os.path.dirname(__file__)


def load_package(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}. Run training first.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # Expect dictionary with keys: model, encoders, feature_cols, target_col
    if not isinstance(obj, dict) or not all(
        k in obj for k in ("model", "encoders", "feature_cols", "target_col")
    ):
        raise ValueError(
            f"{path} missing required keys (model, encoders, feature_cols, target_col)"
        )
    return obj["model"], obj["encoders"], obj["feature_cols"], obj["target_col"]


# Load both packages
prematch_model, prematch_encoders, PREMATCH_FEATURES, PREMATCH_TARGET = load_package(
    os.path.join(BASE, "prematch_model.pkl")
)
midmatch_model, midmatch_encoders, MIDMATCH_FEATURES, MIDMATCH_TARGET = load_package(
    os.path.join(BASE, "midmatch_model.pkl")
)


def transform_with_encoder(encoders, col, value):
    if value is None:
        return None
    if col in encoders:
        try:
            return int(encoders[col].transform([str(value)])[0])
        except Exception:
            return None
    # otherwise try numeric
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except:
        return None


def inverse_label(encoders, target_col, code):
    if target_col in encoders:
        return encoders[target_col].inverse_transform([int(code)])[0]
    return str(code)


def get_dropdowns():
    # Pull dropdown lists from encoders to ensure exact same strings
    teams = (
        list(prematch_encoders["batting_team"].classes_)
        if "batting_team" in prematch_encoders
        else []
    )
    venues = (
        list(prematch_encoders["venue"].classes_)
        if "venue" in prematch_encoders
        else []
    )
    cities = (
        list(prematch_encoders["city"].classes_) if "city" in prematch_encoders else []
    )
    seasons = (
        list(prematch_encoders["season"].classes_)
        if "season" in prematch_encoders
        else []
    )
    match_types = (
        list(prematch_encoders["match_type"].classes_)
        if "match_type" in prematch_encoders
        else []
    )
    toss_decisions = ["bat", "field"]
    return teams, venues, cities, seasons, match_types, toss_decisions


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/prematch")
def prematch():
    teams, venues, cities, seasons, match_types, toss_decisions = get_dropdowns()
    return render_template(
        "prematch.html",
        teams=teams,
        venues=venues,
        cities=cities,
        seasons=seasons,
        match_types=match_types,
        toss_decisions=toss_decisions,
    )


@app.route("/midmatch")
def midmatch():
    teams, venues, cities, _, _, _ = get_dropdowns()
    return render_template("midmatch.html", teams=teams, venues=venues, cities=cities)


@app.route("/predict_prematch", methods=["POST"])
def predict_prematch():
    raw = {col: request.form.get(col) for col in PREMATCH_FEATURES}
    transformed = []
    for col in PREMATCH_FEATURES:
        code = transform_with_encoder(prematch_encoders, col, raw.get(col))
        if code is None:
            return render_template(
                "result.html",
                prediction=f"Invalid or unseen value for '{col}': '{raw.get(col)}'. Please choose from dropdown.",
            )
        transformed.append(code)
    X = np.array([transformed], dtype=float)
    pred_code = prematch_model.predict(X)[0]
    winner = inverse_label(prematch_encoders, PREMATCH_TARGET, pred_code)
    return render_template("result.html", prediction=winner)


@app.route("/predict_midmatch", methods=["POST"])
def predict_midmatch():
    # categories
    batting_team = request.form.get("batting_team")
    bowling_team = request.form.get("bowling_team")
    venue = request.form.get("venue")
    city = request.form.get("city")

    # numerics (runs/wickets/overs or team_balls)
    runs = request.form.get("runs_total") or request.form.get("team_runs")
    wickets = request.form.get("team_wicket")
    overs_input = request.form.get("overs")
    team_balls = request.form.get("team_balls")

    # determine overs
    if overs_input and overs_input.strip() != "":
        try:
            overs = float(overs_input)
        except:
            return render_template("result.html", prediction="Invalid overs value")
    elif team_balls and team_balls.strip() != "":
        try:
            overs = int(team_balls) / 6.0
        except:
            return render_template("result.html", prediction="Invalid team_balls value")
    else:
        return render_template("result.html", prediction="Provide overs or team_balls")

    # encode categories
    encoded = []
    for col, val in [
        ("batting_team", batting_team),
        ("bowling_team", bowling_team),
        ("venue", venue),
        ("city", city),
    ]:
        code = transform_with_encoder(midmatch_encoders, col, val)
        if code is None:
            return render_template(
                "result.html",
                prediction=f"Invalid or unseen value for '{col}': '{val}'. Please choose from dropdown.",
            )
        encoded.append(code)

    # numeric casting
    try:
        team_runs = float(runs)
        team_wicket = int(wickets)
    except:
        return render_template(
            "result.html", prediction="Invalid numeric input for runs or wickets"
        )

    X = np.array([encoded + [team_runs, team_wicket, overs]], dtype=float)
    pred_code = midmatch_model.predict(X)[0]
    winner = inverse_label(midmatch_encoders, MIDMATCH_TARGET, pred_code)
    return render_template("result.html", prediction=winner)


# ✅ New Analysis Route
@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


if __name__ == "__main__":
    app.run(debug=True)
