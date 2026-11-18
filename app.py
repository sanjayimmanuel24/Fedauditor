import os
import uuid

import numpy as np
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    abort,
    request,
    send_from_directory,
    session,
    url_for,
)

from client_valuation import ClientValuation
from config import DECISION_THRESHOLD, DEFAULT_WARMUP_ROUNDS, FEATURE_LABELS, SECRET_KEY
from database import create_user, init_db, verify_user
from experiment_store import (
    append_round,
    create_run,
    init_store,
    rounds_csv,
    save_calibration,
    save_explanation,
    save_fairness,
    save_valuation,
)
from federated_engine import FederatedEngine

app = Flask(__name__)
app.secret_key = os.environ.get("FEDHEALTH_SECRET_KEY", SECRET_KEY)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    JSON_SORT_KEYS=False,
)

init_db()
init_store()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def json_safe(value):
    """Recursively convert numpy scalars/arrays into JSON-friendly types."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Per-user experiment sessions
# ---------------------------------------------------------------------------
class RunSession:
    """One reproducible federated experiment.

    State used to live in a module-level global, which meant every clinician
    shared the same simulation. Runs are now keyed by id and persisted round by
    round, so results survive a restart and can be exported.
    """

    def __init__(self, warmup_rounds: int = DEFAULT_WARMUP_ROUNDS, partition_mode=None):
        self.id = uuid.uuid4().hex
        self.engine = FederatedEngine(partition_mode=partition_mode)
        self.coef, self.intercept = self.engine.initial_parameters()
        self.rounds: list[dict] = []
        self.valuation_report: dict | None = None

        summary = self.engine.summary()
        self.db_run_id = create_run(
            name=f"run-{self.id[:8]}",
            config=summary["hyperparameters"],
            data_source=summary["data_source"],
            partition_mode=summary["partition_mode"],
        )

        for _ in range(int(warmup_rounds)):
            self.step()

    def step(self) -> dict:
        result = self.engine.run_round(
            self.coef, self.intercept, round_no=len(self.rounds) + 1
        )
        self.coef = result["new_coef"]
        self.intercept = result["new_intercept"]

        payload = {
            "round": result["round"],
            "metrics": json_safe(result["metrics"]),
            "local_accuracies": json_safe(result["local_accuracies"]),
            "local_meta": json_safe(result["local_meta"]),
            "confusion_matrix": json_safe(result["confusion_matrix"]),
            "calibration": json_safe(result["calibration"]),
            "fairness": json_safe(result["fairness"]),
            "explanation": json_safe(result["explanation"]),
            "coef": json_safe(result["coef"]),
            "intercept": json_safe(result["intercept"]),
        }
        self.rounds.append(payload)

        append_round(
            self.db_run_id,
            payload["round"],
            payload["metrics"],
            payload["local_accuracies"],
            payload["confusion_matrix"],
        )
        save_calibration(self.db_run_id, payload["round"], payload["calibration"])
        save_fairness(self.db_run_id, payload["round"], payload["fairness"])
        save_explanation(self.db_run_id, payload["round"], payload["explanation"])
        return payload

    @property
    def latest(self) -> dict | None:
        return self.rounds[-1] if self.rounds else None

    def valuation(self, force: bool = False) -> dict:
        if self.valuation_report is None or force:
            report = ClientValuation(self.engine).report()
            self.valuation_report = json_safe(report)
            save_valuation(
                self.db_run_id,
                len(self.rounds),
                self.valuation_report,
            )
        return self.valuation_report

    def state(self) -> dict:
        latest = self.latest
        return {
            "run_id": self.id[:8],
            "db_run_id": self.db_run_id,
            "summary": json_safe(self.engine.summary()),
            "rounds": [
                {
                    "round": entry["round"],
                    "metrics": entry["metrics"],
                    "local_accuracies": entry["local_accuracies"],
                    "local_meta": entry["local_meta"],
                    "confusion_matrix": entry["confusion_matrix"],
                }
                for entry in self.rounds
            ],
            "latest": (
                {
                    "calibration": latest["calibration"],
                    "fairness": latest["fairness"],
                    "explanation": latest["explanation"],
                }
                if latest
                else None
            ),
            "centralised": json_safe(self.engine.centralised_baseline()),
            "has_valuation": self.valuation_report is not None,
        }


RUNS: dict[str, RunSession] = {}


def current_run(create: bool = True) -> RunSession | None:
    run_id = session.get("run_id")
    if run_id and run_id in RUNS:
        return RUNS[run_id]
    if not create:
        return None
    run = RunSession(warmup_rounds=DEFAULT_WARMUP_ROUNDS)
    RUNS[run.id] = run
    session["run_id"] = run.id
    return run


def _require_login():
    return "user" not in session


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.context_processor
def inject_user():
    return dict(user=session.get("user"))


@app.route("/")
@app.route("/landing")
def landing():
    return render_template("landing.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = verify_user(email, password)
        if user:
            session["user"] = user
            flash("Successfully logged in!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif create_user(name, email, password):
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        else:
            flash("Email already registered.", "error")
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("run_id", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("landing"))


@app.route("/dashboard")
def dashboard():
    if _require_login():
        return redirect(url_for("login"))
    run = current_run()
    return render_template(
        "dashboard.html",
        run_id=run.id[:8],
        summary=json_safe(run.engine.summary()),
    )


@app.route("/predict")
def predict_page():
    if _require_login():
        return redirect(url_for("login"))
    run = current_run()
    return render_template(
        "predict.html",
        summary=json_safe(run.engine.summary()),
        feature_labels=FEATURE_LABELS,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@app.route("/predict_data", methods=["POST"])
def predict_data():
    if _require_login():
        return jsonify({"error": "Unauthorized"}), 401

    run = current_run()
    raw = request.get_json(silent=True) or {}
    expected = run.engine.feature_names
    try:
        values = {key: float(raw.get(key, 0.0)) for key in expected}
    except (TypeError, ValueError):
        return jsonify({"error": "All inputs must be numeric."}), 400

    if any(not np.isfinite(v) for v in values.values()):
        return jsonify({"error": "Inputs must be finite numbers."}), 400
    if values["Age"] <= 0:
        return jsonify({"error": "Age must be positive."}), 400

    background = "local" if raw.get("background") == "local" else "global"
    result = run.engine.predict(
        values,
        run.coef,
        run.intercept,
        background=background,
        threshold=DECISION_THRESHOLD,
    )
    return jsonify(json_safe(result))


# ---------------------------------------------------------------------------
# Federated simulation API
# ---------------------------------------------------------------------------
@app.route("/api/state")
def api_state():
    if _require_login():
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(current_run().state())


@app.route("/simulate_round", methods=["POST"])
def simulate_round():
    if _require_login():
        return jsonify({"error": "Unauthorized"}), 401
    run = current_run()
    payload = run.step()
    return jsonify(
        json_safe(
            {
                "round": payload["round"],
                "metrics": payload["metrics"],
                "local_accuracies": payload["local_accuracies"],
                "local_meta": payload["local_meta"],
                "confusion_matrix": payload["confusion_matrix"],
                "calibration": payload["calibration"],
                "fairness": payload["fairness"],
                "explanation": payload["explanation"],
            }
        )
    )


@app.route("/reset_simulation", methods=["POST"])
def reset_simulation():
    if _require_login():
        return jsonify({"error": "Unauthorized"}), 401
    # A genuine cold start: zero weights, zero rounds.
    run = RunSession(warmup_rounds=0)
    RUNS[run.id] = run
    session["run_id"] = run.id
    return jsonify({"status": "success", "run_id": run.id[:8], "rounds": 0})


@app.route("/api/valuation")
def api_valuation():
    if _require_login():
        return jsonify({"error": "Unauthorized"}), 401
    run = current_run()
    force = request.args.get("refresh") == "1"
    return jsonify(run.valuation(force=force))


@app.route("/api/run/<int:run_db_id>/rounds.csv")
def export_rounds(run_db_id: int):
    if _require_login():
        return redirect(url_for("login"))
    payload = rounds_csv(run_db_id)
    return Response(
        payload,
        mimetype="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=fedhealth_rounds_{run_db_id}.csv"
        },
    )


# ---------------------------------------------------------------------------
# FedAuditor: audit report with each claim linked to the tool call it cites
# ---------------------------------------------------------------------------
@app.route("/fedauditor")
def fedauditor_page():
    if _require_login():
        return redirect(url_for("login"))
    from fedauditor.audit_view import DEFAULT_AUDIT, list_audits, load_view

    audits = list_audits()
    name = request.args.get("audit") or (DEFAULT_AUDIT if DEFAULT_AUDIT in audits else
                                        (audits[0] if audits else ""))
    try:
        view = load_view(name)
    except KeyError:
        if request.args.get("audit"):
            abort(404)
        view = None
    figs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fedauditor", "figures")
    figures = [f for f in ("recall_heatmap_dark.svg", "type_confusion_dark.svg") if os.path.exists(os.path.join(figs, f))]
    return render_template("fedauditor.html", audits=audits, view=view, figures=figures)


@app.route("/fedauditor/figures/<path:name>")
def fedauditor_figure(name: str):
    if _require_login():
        return redirect(url_for("login"))
    if name not in ("recall_heatmap_dark.svg", "type_confusion_dark.svg", "recall_heatmap.svg", "type_confusion.svg"):
        abort(404)
    figs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fedauditor", "figures")
    return send_from_directory(figs, name, mimetype="image/svg+xml")


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
        port=int(os.environ.get("PORT", "5000")),
    )
