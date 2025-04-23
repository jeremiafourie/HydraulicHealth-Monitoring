import io
import base64
import pandas as pd
import joblib
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# Class distribution details for descriptions
CLASS_DIST = {
    'cooler_pct': {3: (732, 'close to total failure'), 20: (732, 'reduced efficiency'), 100: (741, 'full efficiency')},
    'valve_pct': {100: (1125, 'optimal switching behavior'), 90: (360, 'small lag'), 80: (360, 'severe lag'), 73: (360, 'close to total failure')},
    'pump_leak': {0: (1221, 'no leakage'), 1: (492, 'weak leakage'), 2: (492, 'severe leakage')},
    'acc_pressure': {130: (599, 'optimal pressure'), 115: (399, 'slightly reduced pressure'), 100: (399, 'severely reduced pressure'), 90: (808, 'close to total failure')}
}

# Feature engineering functions
from src.preprocess_data import build_summary_features, build_fft_features

# Model file paths
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_FILES = {name: ARTIFACTS_DIR / f"rf_{name}.pkl" for name in CLASS_DIST}

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H2("Condition Monitoring of Hydraulic Systems"),
    html.P("Upload raw merged cycles CSV (`hydraulic_cycles.csv`) with sensor columns only:"),
    dcc.Upload(id='upload-data', children=html.Div(['Drag & drop or click to upload']), style={
        'width':'100%','height':'60px','lineHeight':'60px','borderWidth':'1px','borderStyle':'dashed','borderRadius':'5px','textAlign':'center'
    }, multiple=False),
    html.Br(),
    html.Div(id='feature-table'),
    html.Div([html.Label('Cycle index:'), dcc.Input(id='cycle-index', type='number', min=0, step=1)]),
    dbc.Button('Predict Cycle', id='predict-btn', color='primary', disabled=True, className='mt-2'),
    dcc.Store(id='stored-features'),
    dbc.Modal([
        dbc.ModalHeader(html.H4("Prediction for Selected Cycle")),
        dbc.ModalBody(html.Div(id='modal-body')),
        dbc.ModalFooter(dbc.Button('Close', id='close-modal', className='ml-auto'))
    ], id='results-modal', size='lg', is_open=False)
], fluid=True)

# Enable predict button only when features stored and index provided
@app.callback(Output('predict-btn','disabled'), [Input('stored-features','data'), Input('cycle-index','value')])
def enable_predict(feats_json, idx):
    return not feats_json or idx is None

# Parse upload and engineer features
@app.callback([Output('stored-features','data'), Output('feature-table','children')], Input('upload-data','contents'))
def parse_and_engineer(contents):
    if not contents:
        return dash.no_update, dash.no_update
    _, b64 = contents.split(',')
    df = pd.read_csv(io.StringIO(base64.b64decode(b64).decode('utf-8')))
    # build features
    sumf = build_summary_features(df)
    fftf = build_fft_features(df, n_bins=5)
    feats = pd.concat([sumf, fftf], axis=1)
    # insert cycle index
    feats.insert(0, 'cycle_index', feats.index)
    feats_json = feats.to_json(date_format='iso', orient='split')
    # preview first 5 rows
    table = dash_table.DataTable(
        data=feats.head(5).to_dict('records'),
        columns=[{'name':c,'id':c} for c in feats.columns],
        page_size=5,
        style_table={'overflowX':'auto'}
    )
    return feats_json, html.Div([html.H5('Feature Preview (first 5 cycles)'), table])

# Predict single cycle and show modal without progress bars or instance counts
@app.callback([Output('results-modal','is_open'), Output('modal-body','children')],
              [Input('predict-btn','n_clicks'), Input('close-modal','n_clicks')],
              [State('stored-features','data'), State('cycle-index','value'), State('results-modal','is_open')],
              prevent_initial_call=True)
def predict_cycle(n_pred, n_close, feats_json, idx, is_open):
    ctx = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if ctx == 'close-modal':
        return False, dash.no_update
    feats = pd.read_json(feats_json, orient='split')
    # validate index
    try:
        idx = int(idx)
    except:
        return False, html.Div("Invalid cycle index.", style={'color':'red'})
    if idx < 0 or idx >= len(feats):
        return False, html.Div(f"Index out of range (0 to {len(feats)-1}).", style={'color':'red'})
    row = feats.iloc[idx]
    # remove cycle_index before prediction
    if 'cycle_index' in feats.columns:
        row = row.drop('cycle_index')
    content = []
    for target, path in MODEL_FILES.items():
        model = joblib.load(path)
        val = int(model.predict(row.values.reshape(1,-1))[0])
        # description only
        desc = CLASS_DIST[target][val][1]
        content.append(html.Div([html.H5(f"{target}: {val} — {desc}")], className='mb-3'))
    return True, html.Div(content)

if __name__=='__main__':
    app.run(debug=True)
