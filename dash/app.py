"""
dash/app.py
────────────
Dash demo application for the Superconductivity Critical Temperature Predictor.

Three panels:
  1. Prediction form   — input features, get a predicted Tc
  2. Prediction history — live chart of all predictions made so far
  3. Drift status       — which features (if any) are drifting from training data

The app calls the FastAPI backend at API_BASE_URL for all data.
"""

from __future__ import annotations

import os
import time

import dash
import requests
from dash import Input, Output, State, callback, dcc, html

# ─── Config ───────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
REFRESH_INTERVAL_MS = 30_000  # 30 seconds

app = dash.Dash(
    __name__,
    title="Superconductivity Tc Predictor",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expose Flask server for Render/gunicorn


# ─── Helpers ──────────────────────────────────────────────────────────────────


def get_api(path: str, timeout: int = 8) -> dict:
    """GET from the FastAPI backend. Returns empty dict on error."""
    try:
        r = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def post_api(path: str, payload: dict, timeout: int = 8) -> dict:
    """POST to the FastAPI backend. Returns empty dict on error."""
    try:
        r = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def build_feature_inputs(example: dict) -> list:
    """Build labelled number inputs from the /predict/example response."""
    features = example.get("features", {})
    inputs = []
    for feat, val in features.items():
        label = feat.replace("_", " ").replace("wtd", "weighted").title()
        inputs.append(
            html.Div(
                [
                    html.Label(
                        label,
                        style={
                            "fontSize": "12px",
                            "color": "#555",
                            "marginBottom": "3px",
                            "display": "block",
                        },
                    ),
                    dcc.Input(
                        id={"type": "feature-input", "index": feat},
                        type="number",
                        value=round(val, 4),
                        step="any",
                        debounce=True,
                        style={
                            "width": "100%",
                            "padding": "6px 8px",
                            "border": "1px solid #ddd",
                            "borderRadius": "6px",
                            "fontSize": "13px",
                        },
                    ),
                ],
                style={"marginBottom": "12px"},
            )
        )
    return inputs


# ─── Layout ───────────────────────────────────────────────────────────────────

CARD_STYLE = {
    "background": "#fff",
    "borderRadius": "12px",
    "border": "1px solid #e8e8e8",
    "padding": "20px 24px",
    "marginBottom": "20px",
}

HEADER_STYLE = {
    "fontSize": "13px",
    "fontWeight": "600",
    "color": "#333",
    "marginBottom": "14px",
    "letterSpacing": "0.02em",
}


def serve_layout() -> html.Div:
    """Called on every page load — fetches fresh feature list from API."""
    example = get_api("/predict/example")
    health = get_api("/health")
    model_info = get_api("/model/info")

    api_ok = bool(health.get("model_loaded"))
    model_type = model_info.get("model_type", "unknown")
    n_features = model_info.get("n_features", 0)
    cv_rmse = model_info.get("cv_rmse", None)

    status_color = "#22c55e" if api_ok else "#ef4444"
    status_text = "API online" if api_ok else "API offline"

    return html.Div(
        [
            dcc.Interval(
                id="refresh-interval",
                interval=REFRESH_INTERVAL_MS,
                n_intervals=0,
            ),
            # ── Header ──────────────────────────────────────────────────────
            html.Div(
                [
                    html.H1(
                        "Superconductivity Tc Predictor",
                        style={
                            "fontSize": "22px",
                            "fontWeight": "600",
                            "color": "#1a1a1a",
                            "margin": "0 0 4px",
                        },
                    ),
                    html.P(
                        "Predict the critical temperature of superconducting materials "
                        "from compositionally-derived elemental features.",
                        style={"fontSize": "14px", "color": "#666", "margin": "0 0 12px"},
                    ),
                    html.Div(
                        [
                            html.Span(
                                "●",
                                style={"color": status_color, "marginRight": "6px"},
                            ),
                            html.Span(
                                status_text,
                                style={"fontSize": "12px", "color": "#666"},
                            ),
                            html.Span(
                                f"  ·  {model_type}  ·  {n_features} features"
                                + (f"  ·  CV RMSE {cv_rmse:.4f}" if cv_rmse else ""),
                                style={"fontSize": "12px", "color": "#999"},
                            ),
                        ]
                    ),
                ],
                style={
                    "borderBottom": "1px solid #eee",
                    "paddingBottom": "16px",
                    "marginBottom": "24px",
                },
            ),
            # ── Main content ─────────────────────────────────────────────────
            html.Div(
                [
                    # Left column — prediction form
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P("Prediction form", style=HEADER_STYLE),
                                    html.P(
                                        "Values pre-filled with training set medians. "
                                        "Edit any field and click Predict.",
                                        style={
                                            "fontSize": "12px",
                                            "color": "#888",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.Div(
                                        build_feature_inputs(example),
                                        id="feature-inputs-container",
                                    ),
                                    html.Button(
                                        "Predict critical temperature",
                                        id="predict-btn",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "padding": "10px",
                                            "background": "#1a1a2e",
                                            "color": "#fff",
                                            "border": "none",
                                            "borderRadius": "8px",
                                            "fontSize": "14px",
                                            "fontWeight": "500",
                                            "cursor": "pointer",
                                            "marginTop": "4px",
                                        },
                                    ),

                                    dcc.Loading(
                                        id="loading-prediction",
                                        type="circle",
                                        children=html.Div(id="prediction-result", style={"marginTop": "16px"})
                                    ),
                                ],
                                style=CARD_STYLE,
                            )
                        ],
                        style={"flex": "0 0 340px", "minWidth": "280px"},
                    ),
                    # Right column — charts and drift
                    html.Div(
                        [
                            # Prediction history chart
                            html.Div(
                                [
                                    html.P("Prediction history", style=HEADER_STYLE),
                                    dcc.Graph(
                                        id="history-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "260px"},
                                    ),
                                ],
                                style=CARD_STYLE,
                            ),
                            # Drift status card
                            html.Div(
                                [
                                    html.P("Feature drift status", style=HEADER_STYLE),
                                    html.Div(id="drift-status"),
                                ],
                                style=CARD_STYLE,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "0"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "20px",
                    "alignItems": "flex-start",
                    "flexWrap": "wrap",
                },
            ),
        ],
        style={
            "maxWidth": "1100px",
            "margin": "0 auto",
            "padding": "28px 20px",
            "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "background": "#f8f9fa",
            "minHeight": "100vh",
        },
    )


app.layout = serve_layout


# ─── Callbacks ────────────────────────────────────────────────────────────────


@callback(
    Output("prediction-result", "children"),
    Input("predict-btn", "n_clicks"),
    State({"type": "feature-input", "index": dash.ALL}, "value"),
    State({"type": "feature-input", "index": dash.ALL}, "id"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, values, ids):
    if not n_clicks:
        return dash.no_update

    features = {
        id_dict["index"]: float(v) if v is not None else 0.0
        for id_dict, v in zip(ids, values, strict=False)
    }

    result = post_api("/predict", {"features": features})

    if "error" in result:
        return html.Div(
            f"Error: {result['error']}",
            style={
                "padding": "12px",
                "background": "#fef2f2",
                "border": "1px solid #fca5a5",
                "borderRadius": "8px",
                "color": "#b91c1c",
                "fontSize": "13px",
            },
        )

    raw = result.get("predicted_critical_temp_boxcox", 0)
    imputed = result.get("imputed_features", [])
    request_id = result.get("request_id", "")
    model_type = result.get("model_type", "")

    # Convert Box-Cox back to Kelvin using approximate inverse
    # (exact conversion requires fitted lambda — this is approximate)
    tc_approx_k = max(0, raw ** (1 / 0.24)) if raw > 0 else 0
    tc_approx_c = tc_approx_k - 273.15

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        f"{raw:.4f}",
                        style={"fontSize": "28px", "fontWeight": "600", "color": "#1a1a2e"},
                    ),
                    html.Span(
                        " Box-Cox scale",
                        style={"fontSize": "13px", "color": "#888", "marginLeft": "6px"},
                    ),
                ],
                style={"marginBottom": "6px"},
            ),
            html.P(
                f"≈ {tc_approx_k:.1f} K  /  {tc_approx_c:.1f} °C  (approximate)",
                style={"fontSize": "13px", "color": "#555", "margin": "0 0 8px"},
            ),
            html.P(
                f"Model: {model_type}  ·  ID: {request_id[:16]}…",
                style={"fontSize": "11px", "color": "#aaa", "margin": "0 0 6px"},
            ),
            html.Div(
                f"Imputed {len(imputed)} missing features: {', '.join(imputed)}"
                if imputed
                else "",
                style={"fontSize": "11px", "color": "#f59e0b"},
            ),
        ],
        style={
            "padding": "14px",
            "background": "#f0f9ff",
            "border": "1px solid #bae6fd",
            "borderRadius": "8px",
        },
    )


@callback(
    Output("history-chart", "figure"),
    Input("refresh-interval", "n_intervals"),
    Input("predict-btn", "n_clicks"),
)
def update_history_chart(n_intervals, n_clicks):
    logs = get_api("/monitoring/logs")
    n = logs.get("n_predictions_logged", 0)
    stats = logs.get("prediction_stats", {})

    if n == 0:
        return {
            "data": [],
            "layout": {
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "annotations": [
                    {
                        "text": "No predictions yet — submit the form to get started",
                        "xref": "paper",
                        "yref": "paper",
                        "showarrow": False,
                        "font": {"size": 13, "color": "#999"},
                        "x": 0.5,
                        "y": 0.5,
                    }
                ],
                "plot_bgcolor": "#fff",
                "paper_bgcolor": "#fff",
            },
        }

    # Box plot of prediction distribution using summary stats
    fig = {
        "data": [
            {
                "type": "box",
                "q1": [stats.get("p25", 0)],
                "median": [stats.get("p50", 0)],
                "q3": [stats.get("p75", 0)],
                "lowerfence": [stats.get("min", 0)],
                "upperfence": [stats.get("max", 0)],
                "mean": [stats.get("mean", 0)],
                "name": "Tc (Box-Cox)",
                "marker": {"color": "#1a1a2e"},
                "boxmean": True,
            }
        ],
        "layout": {
            "height": 220,
            "margin": {"l": 40, "r": 20, "t": 20, "b": 40},
            "yaxis": {"title": "Predicted Tc (Box-Cox scale)", "titlefont": {"size": 11}},
            "plot_bgcolor": "#fff",
            "paper_bgcolor": "#fff",
            "annotations": [
                {
                    "text": f"n = {n} predictions  ·  mean = {stats.get('mean', 0):.3f}",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 11, "color": "#888"},
                    "x": 0.5,
                    "y": -0.18,
                    "xanchor": "center",
                }
            ],
        },
    }
    return fig


@callback(
    Output("drift-status", "children"),
    Input("refresh-interval", "n_intervals"),
    Input("predict-btn", "n_clicks"),
)
def update_drift_status(n_intervals, n_clicks):
    drift = get_api("/monitoring/drift")

    if not drift:
        return html.P(
            "Drift monitor unavailable — reference data not found.",
            style={"fontSize": "13px", "color": "#888"},
        )

    report = drift.get("drift_report", {})
    drifted = drift.get("drifted_features", [])
    feature_results = report.get("feature_results", {})

    if not feature_results:
        return html.P(
            "No drift data yet.",
            style={"fontSize": "13px", "color": "#888"},
        )

    overall_ok = len(drifted) == 0
    overall_color = "#22c55e" if overall_ok else "#ef4444"
    overall_text = "No drift detected" if overall_ok else f"{len(drifted)} feature(s) drifting"

    rows = []
    for feat, vals in sorted(
        feature_results.items(),
        key=lambda x: x[1].get("p_value", 1),
    ):
        is_drifted = vals.get("drifted", False)
        p_val = vals.get("p_value", 1)
        ks_stat = vals.get("ks_statistic", 0)
        dot_color = "#ef4444" if is_drifted else "#22c55e"
        rows.append(
            html.Div(
                [
                    html.Span("●", style={"color": dot_color, "marginRight": "8px"}),
                    html.Span(
                        feat.replace("_", " "),
                        style={"fontSize": "12px", "flex": "1", "color": "#333"},
                    ),
                    html.Span(
                        f"KS={ks_stat:.3f}  p={p_val:.4f}",
                        style={"fontSize": "11px", "color": "#999", "fontFamily": "monospace"},
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": "5px 0",
                    "borderBottom": "1px solid #f0f0f0",
                },
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span("●", style={"color": overall_color, "fontSize": "16px", "marginRight": "8px"}),
                    html.Span(
                        overall_text,
                        style={"fontSize": "14px", "fontWeight": "600", "color": overall_color},
                    ),
                ],
                style={"marginBottom": "12px"},
            ),
            html.Div(rows),
            html.P(
                "Comparing prediction log vs training reference data (KS-test, α=0.05)",
                style={"fontSize": "11px", "color": "#bbb", "marginTop": "10px"},
            ),
        ]
    )


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)