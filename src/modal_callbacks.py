import io
import base64
import pandas as pd
import joblib
from pathlib import Path
import os
import sys

import dash_bootstrap_components as dbc

import dash
from dash import Input, Output, State, html
import dash_bootstrap_components as dbc
from dash import dash_table

# Class distribution details for descriptions
CLASS_DIST = {
    'cooler_pct': {
        3:   (732, 'close to total failure'),
        20:  (732, 'reduced efficiency'),
        100: (741, 'full efficiency')
    },
    'valve_pct': {
        100: (1125, 'optimal switching behavior'),
        90:   (360, 'small lag'),
        80:   (360, 'severe lag'),
        73:   (360, 'close to total failure')
    },
    'pump_leak': {
        0:   (1221, 'no leakage'),
        1:    (492, 'weak leakage'),
        2:    (492, 'severe leakage')
    },
    'acc_pressure': {
        130: (599, 'optimal pressure'),
        115: (399, 'slightly reduced pressure'),
        100: (399, 'severely reduced pressure'),
        90:  (808, 'close to total failure')
    }
}

# Where models are stored
dir_path = Path(__file__).parent.parent / 'artifacts'
MODEL_FILES = {name: dir_path / f"rf_{name}.pkl" for name in CLASS_DIST}

# Feature engineering helpers

from src.preprocess_data import build_summary_features, build_fft_features


def register_modal_callbacks(app):
    """
    Registers the single-cycle prediction modal callback with the given Dash app.
    """
    @app.callback(
        [Output('results-modal', 'is_open'), Output('modal-body', 'children')],
        [Input('predict-btn', 'n_clicks'), Input('close-modal', 'n_clicks')],
        [State('stored-features', 'data'), State('cycle-index', 'value'), State('results-modal', 'is_open')],
        prevent_initial_call=True
    )
    def predict_cycle(n_predict, n_close, feats_json, idx, is_open):
        ctx = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        if ctx == 'close-modal':
            return False, dash.no_update

        # Load engineered features
        feats = pd.read_json(io.StringIO(feats_json), orient='split')
        # Validate cycle index
        try:
            idx = int(idx)
        except:
            return False, html.Div("Invalid cycle index.", style={'color':'red'})
        if idx < 0 or idx >= len(feats):
            return False, html.Div(f"Index out of range (0 to {len(feats)-1}).", style={'color':'red'})

        # Select row and drop index column if present
        row = feats.iloc[idx]
        if 'cycle_index' in feats.columns:
            row = row.drop('cycle_index')

        # Run each model and build descriptive output
        content = []
        emoji = None  # Initialize emoji variable
        descbody = None
        for target, path in MODEL_FILES.items():
            model = joblib.load(path)
            val = int(model.predict(row.values.reshape(1, -1))[0])
            desc = CLASS_DIST[target][val][1]

            match target:
                case 'cooler_pct':
                    match val:
                        case v if  v < 4:
                            emoji = "‼️"
                            descbody = "Urgent! Replace or repair cooler immediately. System risk is high."
                        case v if 3 < v < 100:
                            emoji = "⚠️"
                            descbody = "Check for blockages or fouling. Consider scheduling maintenance."
                        case v if 99 < v:
                            emoji = "✅"
                            descbody = " No action needed. Keep monitoring."
                
                case 'valve_pct':
                    match val:
                        case v if v < 74:
                            emoji = "❌"
                            descbody = "Immediate replacement recommended. High system failure risk."
                        case v if 73 < v < 81:
                            emoji = "❗"
                            descbody = "Inspect valve controls and possible actuator issues. Plan for repair."
                        case v if 80 < v < 100:
                            emoji = "⚠️"
                            descbody = "Monitor performance. Check electrical signals and valve response time."
                        case v if 99 < v:
                            emoji = "✅"
                            descbody = "All good. No intervention required."

                case 'pump_leak':
                     match val:
                        case 2:
                            emoji = "❌"
                            descbody = "Repair or replace seals/pump immediately."
                        case 1:
                            emoji = "⚠️"
                            descbody = "Monitor seal wear. Schedule preventive maintenance."
                        case 0:
                            emoji = "✅"
                            descbody = "All good. Maintain current schedule."
                       

                case 'acc_pressure':
                     match val:
                        case v if v < 91:
                            emoji = "❌"
                            descbody = "Replace accumulator. Pressure too low for safe operation."
                        case v if 90 < v < 101:
                            emoji = "❗"
                            descbody = "Recharge nitrogen or replace bladder. Performance degradation likely."
                        case v if 100 < v < 116:
                            emoji = "⚠️"
                            descbody = "Check pre-charge pressure. Watch for early signs of gas leakage"
                        case v if 129 < v:
                            emoji = "✅"
                            descbody = "System pressure normal. No action needed."

                        
                
                    

            content.append(
                html.Div([ 
                    dbc.Card([
                        dbc.CardBody([

                            html.H4(f"{emoji}{target}: {val} — {desc}", className="card-title"),
                            html.P(f"{descbody}",className="card-text")
                        ])
                    ]),
                    # html.H5(f"{emoji}{target}: {val} — {desc}"),
                    
                ], className='mb-3')
            )
        return True, html.Div(content)
