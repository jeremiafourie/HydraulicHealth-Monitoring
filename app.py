
import sys
print("Python executable:", sys.executable)
print("sys.path[0]:", sys.path[0])
import dash  # this will still error if dash isn’t on this PYTHONPATH

import os
import io
import base64
import pandas as pd
import joblib
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Paths to pre-trained models in artifacts/
ARTIFACTS_DIR = Path(__file__).parent / 'artifacts'
MODEL_FILES = {
    'cooler_pct':   ARTIFACTS_DIR / 'rf_cooler_pct.pkl',
    'valve_pct':    ARTIFACTS_DIR / 'rf_valve_pct.pkl',
    'pump_leak':    ARTIFACTS_DIR / 'rf_pump_leak.pkl',
    'acc_pressure': ARTIFACTS_DIR / 'rf_acc_pressure.pkl'
}
# Corresponding logos in assets/
LOGO_FILES = {
    'cooler_pct':   'assets/cooler_logo.png',
    'valve_pct':    'assets/valve_logo.png',
    'pump_leak':    'assets/pump_logo.png',
    'acc_pressure': 'assets/acc_logo.png'
}

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout
app.layout = dbc.Container([
    html.H1("Hydraulic System Condition Monitor", className="my-4"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'marginBottom': '20px'
        },
        multiple=False
    ),
    dbc.Button("Run Predictions", id='predict-btn', color='primary', disabled=True),

    # Hidden store for uploaded DataFrame
    dcc.Store(id='stored-data'),

    # Modal for progress bars
    dbc.Modal([
        dbc.ModalHeader(html.H4("Prediction Results")),
        dbc.ModalBody(id='modal-body'),
        dbc.ModalFooter(
            dbc.Button("Close", id='close-modal', className="ml-auto")
        ),
    ], id='results-modal', size='lg', is_open=False)
], fluid=True)

# Callback: parse upload, enable predict button
@app.callback(
    Output('stored-data', 'data'),
    Output('predict-btn', 'disabled'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def parse_upload(contents, filename):
    if contents is None:
        return dash.no_update, True
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df.to_json(date_format='iso', orient='split'), False

# Callback: run predictions and show modal
@app.callback(
    Output('results-modal', 'is_open'),
    Output('modal-body', 'children'),
    Input('predict-btn', 'n_clicks'),
    Input('close-modal', 'n_clicks'),
    State('stored-data', 'data'),
    State('results-modal', 'is_open')
)
def show_results(run_click, close_click, data_json, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'close-modal':
        return False, dash.no_update
    # Else, run predictions
    df = pd.read_json(data_json, orient='split')
    bars = []
    for target, model_path in MODEL_FILES.items():
        # load model and predict
        model = joblib.load(model_path)
        preds = model.predict(df)
        # compute distribution of predicted classes
        counts = pd.Series(preds).value_counts(normalize=True)
        # build progress bar segments
        segments = []
        for cls, frac in counts.items():
            segments.append({
                'label': f"{int(cls)}", 'value': int(frac*100), 'color': 'info'
            })
        # container with logo and progress bar
        bars.append(
            dbc.Row([
                dbc.Col(html.Img(src=LOGO_FILES[target], height='40px'), width='auto'),
                dbc.Col(
                    dbc.Progress(segments, multi=True, style={'height': '40px'}), width=10
                )
            ], align='center', className='mb-3')
        )
    return True, bars

if __name__ == '__main__':
    app.run_server(debug=True)
