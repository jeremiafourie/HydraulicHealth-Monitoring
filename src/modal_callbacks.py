import io
import base64
import pandas as pd
import joblib
from pathlib import Path

import dash
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc

# Class distribution details for descriptions
CLASS_DIST = {
    'cooler_pct': {3: (732, 'close to total failure'), 20: (732, 'reduced efficiency'), 100: (741, 'full efficiency')},
    'valve_pct': {100: (1125, 'optimal switching behavior'), 90: (360, 'small lag'), 80: (360, 'severe lag'), 73: (360, 'close to total failure')},
    'pump_leak': {0: (1221, 'no leakage'), 1: (492, 'weak leakage'), 2: (492, 'severe leakage')},
    'acc_pressure': {130: (599, 'optimal pressure'), 115: (399, 'slightly reduced pressure'), 100: (399, 'severely reduced pressure'), 90: (808, 'close to total failure')}
}

# Where models are stored
dir_path = Path(__file__).parent.parent / 'artifacts'
MODEL_FILES = {name: dir_path / f"rf_{name}.pkl" for name in CLASS_DIST}

# Feature engineering helpers
from src.preprocess_data import build_summary_features, build_fft_features


def register_modal_callbacks(app):
    """
    Registers the modal open/close toggle and the prediction content callback.
    """
    # 1) Toggle modal open/close immediately when button clicked
    @app.callback(
        Output('results-modal', 'is_open'),
        [Input('predict-btn', 'n_clicks'), Input('close-modal', 'n_clicks')],
        [State('results-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_modal(n_predict, n_close, is_open):
        ctx = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        if ctx == 'predict-btn':
            return True
        if ctx == 'close-modal':
            return False
        return is_open

    # 2) Compute and update modal-body when predict button clicked
    @app.callback(
        Output('modal-body', 'children'),
        [Input('predict-btn', 'n_clicks')],
        [State('stored-features', 'data'), State('cycle-index', 'value')],
        prevent_initial_call=True
    )
    def predict_cycle(n_predict, feats_json, idx):
        # Load engineered features
        feats = pd.read_json(io.StringIO(feats_json), orient='split')
        # Validate cycle index
        try:
            idx = int(idx)
        except:
            return html.Div("Invalid cycle index.", style={'color': 'red'})
        if idx < 0 or idx >= len(feats):
            return html.Div(f"Index out of range (0 to {len(feats)-1}).", style={'color': 'red'})

        # Select row and drop index column if present
        row = feats.iloc[idx]
        if 'cycle_index' in feats.columns:
            row = row.drop('cycle_index')

        # Run each model and build descriptive output
        content = []
        for target, path in MODEL_FILES.items():
            model = joblib.load(path)
            val = int(model.predict(row.values.reshape(1, -1))[0])
            desc = CLASS_DIST[target][val][1]
            # Determine emoji and description body
            if target == 'cooler_pct':
                if val < 4:
                    emoji, descbody = "‼️", "Urgent! Replace or repair cooler immediately. System risk is high."
                elif val < 100:
                    emoji, descbody = "⚠️", "Check for blockages or fouling. Consider scheduling maintenance."
                else:
                    emoji, descbody = "✅", "No action needed. Keep monitoring."

            elif target == 'valve_pct':
                if val < 74:
                    emoji, descbody = "❌", "Immediate replacement recommended. High system failure risk."
                elif val < 81:
                    emoji, descbody = "❗", "Inspect valve controls and possible actuator issues. Plan for repair."
                elif val < 100:
                    emoji, descbody = "⚠️", "Monitor performance. Check electrical signals and valve response time."
                else:
                    emoji, descbody = "✅", "All good. No intervention required."

            elif target == 'pump_leak':
                if val == 2:
                    emoji, descbody = "❌", "Repair or replace seals/pump immediately."
                elif val == 1:
                    emoji, descbody = "⚠️", "Monitor seal wear. Schedule preventive maintenance."
                else:
                    emoji, descbody = "✅", "All good. Maintain current schedule."

            elif target == 'acc_pressure':
                if val < 91:
                    emoji, descbody = "❌", "Replace accumulator. Pressure too low for safe operation."
                elif val < 101:
                    emoji, descbody = "❗", "Recharge nitrogen or replace bladder. Performance degradation likely."
                elif val < 116:
                    emoji, descbody = "⚠️", "Check pre-charge pressure. Watch for early signs of gas leakage."
                else:
                    emoji, descbody = "✅", "System pressure normal. No action needed."

            # Build card for each target
            content.append(
                dbc.Card(
                    dbc.CardBody([
                        html.H4(f"{emoji} {target}: {val} — {desc}", className="card-title"),
                        html.P(descbody, className="card-text")
                    ]),
                    className='mb-3 shadow-sm'
                )
            )

        return html.Div(content)
