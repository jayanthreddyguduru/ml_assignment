"""BITS ML Assignment 2: train and assess five models on UCI Bank Marketing.
Run inside BITS Virtual Lab: python train_models.py
"""
from pathlib import Path
import json
import zipfile
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODEL = ROOT / 'model'
URL = 'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip'
SEED = 42
EXCLUDE_DURATION = True  # no pre-call access to duration; avoids target leakage


def load_data():
    DATA.mkdir(exist_ok=True)
    path = DATA / 'bank-additional-full.csv'
    if not path.exists():
        outer = DATA / 'bank+marketing.zip'
        print('Downloading original UCI dataset...')
        urlretrieve(URL, outer)
        with zipfile.ZipFile(outer) as z:
            embedded = next(n for n in z.namelist() if n.endswith('bank-additional.zip'))
            with zipfile.ZipFile(__import__('io').BytesIO(z.read(embedded))) as inner:
                name = next(n for n in inner.namelist() if n.endswith('bank-additional-full.csv'))
                path.write_bytes(inner.read(name))
    df = pd.read_csv(path, sep=';')
    assert df.shape == (41188, 21), f'Unexpected input shape: {df.shape}'
    assert set(df['y'].unique()) == {'yes', 'no'}
    return df


def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]  # positive class is 1
    return {
        'Accuracy': accuracy_score(y_test, pred),
        'AUC': roc_auc_score(y_test, scores),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, pred),
    }


def main():
    MODEL.mkdir(exist_ok=True)
    df = load_data()
    features = df.drop(columns=['y'] + (['duration'] if EXCLUDE_DURATION else []))
    target = df['y'].map({'no': 0, 'yes': 1})
    # Ensure same holdout for all five models; keep it untouched while fitting.
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=SEED, stratify=target)
    out_test = x_test.copy()
    out_test['y'] = y_test.map({0: 'no', 1: 'yes'})
    out_test.to_csv(ROOT / 'test_data.csv', index=False)

    cat = features.select_dtypes(include=['object', 'category']).columns.tolist()
    num = features.select_dtypes(exclude=['object', 'category']).columns.tolist()
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), num),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
    ], sparse_threshold=0)

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1500, random_state=SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED),
    }
    rows = []
    manifest = {}
    for name, estimator in classifiers.items():
        print(f'Training {name}...', flush=True)
        pipeline = Pipeline([('preprocess', __import__('sklearn.base', fromlist=['clone']).clone(preprocess)),
                             ('model', estimator)])
        pipeline.fit(x_train, y_train)
        scores = evaluate(pipeline, x_test, y_test)
        rows.append({'ML Model Name': name, **scores})
        filename = name.lower().replace(' ', '_') + '.joblib'
        joblib.dump(pipeline, MODEL / filename, compress=3)
        manifest[name] = filename
        print({k: round(v, 4) for k, v in scores.items()})

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / 'model_comparison.csv', index=False)
    (MODEL / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (MODEL / 'metadata.json').write_text(json.dumps({
        'dataset': 'UCI Bank Marketing / bank-additional-full.csv',
        'dataset_rows': len(df), 'original_features': 20,
        'trained_features': features.shape[1], 'excluded_duration': EXCLUDE_DURATION,
        'train_rows': len(x_train), 'test_rows': len(x_test),
        'random_state': SEED, 'test_fraction': 0.2,
        'features': features.columns.tolist(), 'target': 'y',
        'positive_class': 'yes'
    }, indent=2), encoding='utf-8')
    print('\nMODEL COMPARISON\n', results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nSaved trained pipelines, model comparison, and test_data.csv.')


if __name__ == '__main__':
    main()
