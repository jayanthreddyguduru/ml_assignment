"""Streamlit evaluation UI: does not fit or retrain any model."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Bank Marketing | ML Model Lab', page_icon='🏦', layout='wide')
st.title('🏦 Bank Marketing — Classification Model Lab')
st.caption('BITS ML Assignment 2 | Evaluate previously trained UCI Bank Marketing classifiers')
st.info('The app loads serialized training pipelines. Test CSVs are evaluated only; **no training occurs here**.')

manifest_file = ROOT / 'model' / 'manifest.json'
metadata_file = ROOT / 'model' / 'metadata.json'
if not manifest_file.exists() or not metadata_file.exists():
    st.error('Model files not found. Run `python train_models.py` locally first and commit the generated model/ folder.')
    st.stop()
manifest = json.loads(manifest_file.read_text())
metadata = json.loads(metadata_file.read_text())

st.sidebar.header('Experiment controls')
selected = st.sidebar.selectbox('Choose a classifier', list(manifest))
show_all = st.sidebar.checkbox('Compare all five models', value=True)
file = st.file_uploader('Upload held-out test CSV (must contain original feature columns and `y`)', type=['csv'])
if file is None:
    sample = ROOT / 'test_data.csv'
    if sample.exists():
        st.download_button('Download sample test CSV', sample.read_bytes(), 'test_data.csv', 'text/csv')
    st.stop()
try:
    frame = pd.read_csv(file)
except Exception as e:
    st.error(f'Cannot read CSV: {e}')
    st.stop()
needed = metadata['features'] + ['y']
missing = [c for c in needed if c not in frame.columns]
if missing:
    st.error('CSV is missing columns: ' + ', '.join(missing))
    st.stop()
if frame.empty:
    st.error('Uploaded CSV is empty.')
    st.stop()
if not set(frame['y'].dropna().astype(str)).issubset({'yes', 'no'}):
    st.error('The target `y` must use values yes/no.')
    st.stop()
if frame['y'].isna().any() or frame['y'].nunique() != 2:
    st.error('AUC needs both positive (yes) and negative (no) true labels in the test CSV.')
    st.stop()
X = frame[metadata['features']].copy()
for c in X.columns:
    if c not in X.select_dtypes(include=['object']).columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
y = frame['y'].map({'no': 0, 'yes': 1})
st.write(f'**Test rows:** {len(frame):,} | **Features used:** {len(metadata["features"])}')

@st.cache_resource(show_spinner=False)
def load_pipeline(filename):
    # Only locally committed, trusted model files are loaded, never user-uploaded pickle files.
    return joblib.load(ROOT / 'model' / filename)

def score(name):
    model = load_pipeline(manifest[name])
    predictions = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    scores = {
        'Model': name,
        'Accuracy': accuracy_score(y, predictions),
        'AUC': roc_auc_score(y, probability),
        'Precision': precision_score(y, predictions, zero_division=0),
        'Recall': recall_score(y, predictions, zero_division=0),
        'F1': f1_score(y, predictions, zero_division=0),
        'MCC': matthews_corrcoef(y, predictions),
    }
    return scores, predictions

try:
    chosen_scores, preds = score(selected)
    st.subheader(f'{selected} — test evaluation')
    cells = st.columns(6)
    for cell, key in zip(cells, ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']):
        cell.metric(key, f'{chosen_scores[key]:.3f}')
    st.subheader('Confusion matrix')
    cm = confusion_matrix(y, preds, labels=[0, 1])
    st.dataframe(pd.DataFrame(cm, index=['Actual no', 'Actual yes'],
                              columns=['Predicted no', 'Predicted yes']))
    st.caption('Positive class: yes. False negatives are actual yes predicted as no.')
    if show_all:
        st.subheader('All classifiers — same uploaded test set')
        comparison = pd.DataFrame([score(name)[0] for name in manifest])
        st.dataframe(comparison.style.format({key: '{:.4f}' for key in comparison.columns if key != 'Model'}),
                     use_container_width=True, hide_index=True)
        st.download_button('Export evaluation table', comparison.to_csv(index=False), 'uploaded_test_metrics.csv')
except Exception as e:
    st.error(f'Evaluation failed: {type(e).__name__}: {e}')
    st.info('Check feature types and confirm models were trained with the supplied requirements.txt.')
