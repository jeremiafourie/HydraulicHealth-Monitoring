import io
import base64
import pandas as pd
import joblib
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Directory where your .pkl models live
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODELS = {
    "cooler_pct":   ARTIFACTS_DIR / "rf_cooler_pct.pkl",
    "valve_pct":    ARTIFACTS_DIR / "rf_valve_pct.pkl",
    "pump_leak":    ARTIFACTS_DIR / "rf_pump_leak.pkl",
    "acc_pressure": ARTIFACTS_DIR / "rf_acc_pressure.pkl",
}

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout: upload + button + results
app.layout = dbc.Container([
    html.H2("Hydraulic System Predictor"),
    html.P("Upload a CSV of engineered features (one row per cycle):"),
    dcc.Upload(
        id="upload-data",
        children=html.Div(["Drag & drop or click to upload"]),
        style={"width":"100%","height":"60px","lineHeight":"60px",
               "borderWidth":"1px","borderStyle":"dashed",
               "borderRadius":"5px","textAlign":"center"},
        multiple=False
    ),
    html.Br(),
    dbc.Button("Run Predictions", id="run-btn", color="primary", disabled=True),
    html.Div(id="results-div", className="mt-4")
], fluid=True)

# Enable the button when a file is uploaded
def enable_button(contents):
    return contents is None
app.callback(Output("run-btn","disabled"), Input("upload-data","contents"))(enable_button)

# Handle predictions
@app.callback(
    Output("results-div","children"),
    Input("run-btn","n_clicks"),
    Input("upload-data","contents"),
    prevent_initial_call=True
)
def run_predictions(n_clicks, contents):
    if not contents:
        return dash.no_update
    # Parse CSV
    try:
        prefix, b64 = contents.split(",")
        decoded = base64.b64decode(b64)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        return html.Div(f"Failed to parse CSV: {e}", style={'color':'red'})
    # Check model files
    missing = [name for name,path in MODELS.items() if not path.exists()]
    if missing:
        return html.Div(
            f"Missing model files: {', '.join(missing)}. Please train and save them in artifacts/.",
            style={'color':'red'}
        )
    # Predict and build progress bars
    output_rows = []
    for name, path in MODELS.items():
        try:
            model = joblib.load(path)
            preds = model.predict(df)
            freqs = pd.Series(preds).value_counts(normalize=True).sort_index()
            # Create stacked bars
            bars = [dbc.Progress(value=int(freq*100), color='info', label=str(int(cls)), bar=True)
                    for cls,freq in freqs.items()]
            stack = dbc.Progress(children=bars, style={'height':'30px'})
            output_rows.append(html.Div([html.B(name), stack], className='mb-3'))
        except Exception as e:
            output_rows.append(html.Div(f"Error with {name}: {e}", style={'color':'red'}))
    return output_rows

if __name__ == '__main__':
    app.run(debug=True)
