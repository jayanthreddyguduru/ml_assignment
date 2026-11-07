# Bank Marketing — Classification Model Lab

**BITS Pilani | M.Tech (AIML) | Machine Learning Assignment 2**  
**Student ID:** 2026AJ05346

## a. Problem statement

Predict whether a customer will subscribe to a bank term deposit (`y = yes` or `no`) based on customer, contact, campaign, and economic attributes. Train and compare five supervised classification models on a common held-out test set, and deploy a Streamlit evaluation interface that loads saved pipelines without retraining.

## b. Dataset description

- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- **Dataset file:** `bank-additional-full.csv` from the Bank Marketing collection.
- **Size:** 41,188 observations, 20 original input attributes, and one binary target (`y`).
- **Target classes:** `yes` (customer subscribed) and `no` (did not subscribe).
- **Features used:** 19; `duration` is excluded because call duration is not available before a marketing call. The original file remains unchanged.
- **Split:** Stratified 80% train (32,950 rows) and 20% test (8,238 rows), `random_state=42`.
- **Preprocessing:** Numeric median imputation plus standard scaling; categorical most-frequent imputation plus one-hot encoding (`handle_unknown='ignore'`). Preprocessing is fitted on training data inside each saved scikit-learn pipeline.
- **Evaluation protocol:** Identical held-out test set for all models; default classifier decision thresholds. AUC is calculated from positive-class probability scores, while the other metrics use class predictions.
- **Caveat:** The original dataset is ordered in time, whereas this assignment uses a random stratified holdout. A later-period holdout would provide an additional deployment-realism check.

## c. GitHub Repository Link

https://github.com/jayanthreddyguduru/ml_assignment

## d. Models used and comparison

The five models are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and RBF Support Vector Machine. Evaluation metrics are calculated using the positive class `yes`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9009 | 0.8008 | 0.6905 | 0.2188 | 0.3322 | 0.3516 |
| Decision Tree | 0.8996 | 0.8061 | 0.6305 | 0.2629 | 0.3711 | 0.3636 |
| KNN | 0.9003 | 0.7756 | 0.6458 | 0.2554 | 0.3660 | 0.3641 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| Support Vector Machine | 0.9035 | 0.7060 | 0.6950 | 0.2554 | 0.3735 | 0.3828 |

### Per-model performance observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | High accuracy (0.9009), strong precision (0.6905), and AUC (0.8008), but low recall (0.2188): it misses many actual subscribers at the default threshold. |
| Decision Tree | Highest AUC (0.8061) of the five models, with modest recall (0.2629) and F1 (0.3711). Its depth and leaf-size constraints help control complexity; the results alone do not establish whether it overfits. |
| KNN | Accuracy (0.9003) is close to Logistic Regression, with recall (0.2554) and F1 (0.3660). Distance-based predictions can be sensitive to the encoded feature space and feature scaling. |
| Naive Bayes | Highest recall (0.6347) and F1 (0.4230), capturing the greatest share of actual subscribers, but at lower precision (0.3172) and accuracy (0.8049). It is attractive when missing a potential subscriber is costly. |
| Support Vector Machine | Highest accuracy (0.9035), precision (0.6950), and MCC (0.3828), but recall is only 0.2554 and AUC is lowest (0.7060). Strongest under a criterion emphasizing MCC and false-positive control, not subscriber coverage. |
| **Overall winner for this dataset** | **Support Vector Machine (SVM)**, when MCC is selected as the primary overall metric: its MCC of **0.3828** is highest, along with the highest accuracy (**0.9035**) and precision (**0.6950**). This is a conditional choice, not a universal winner. **Naive Bayes** would be preferable if the business objective prioritizes finding as many potential subscribers as possible, since its recall (**0.6347**) and F1 (**0.4230**) are best. |

### Interpretation and limitations

The data are class-imbalanced. A high accuracy can conceal low identification of the positive (`yes`) class. For example, Logistic Regression correctly detects only 203 of 928 positive cases in the 8,238-row test set (recall 0.2188). Threshold tuning, precision-recall curves, temporal evaluation, and explicit business costs would be worthwhile follow-up experiments. No hyperparameter tuning or cross-validation is claimed here. No causal conclusions should be drawn from these models.

## Streamlit application

**Live link:** https://ml-assignment-jayanth.streamlit.app/

The deployed application supports CSV upload of a labeled test dataset, classifier selection, six evaluation metrics, a confusion matrix, and a side-by-side comparison of the five models. It loads saved `.joblib` preprocessing/classifier pipelines and does not fit models on uploaded test data. The uploaded CSV must include the expected original input fields used in training and target column `y`.

## How to reproduce

1. Use Python 3.12 (or a compatible Python installation).
2. Install the dependencies from `requirements.txt` (ensure scikit-learn 1.7.2 for loading the published model artifacts).
3. Run `python train_models.py` to train and export the models and test data.
4. Run `python -m streamlit run app.py` to start the interactive app.
5. Upload the generated `test_data.csv` in the app and compare models.

The models were also trained and evaluated in the BITS Virtual Lab; the execution screenshot is included in the assignment submission PDF.

## Repository files

- `train_models.py` — dataset loading, preprocessing, training, evaluation, model saving.
- `app.py` — Streamlit interface for evaluating trained pipelines.
- `requirements.txt` — runtime dependencies.
- `test_data.csv` — held-out labeled test dataset for evaluation.
- `model_comparison.csv` — metric results for all five models.
- `model/*.joblib` — five saved fitted preprocessing and classification pipelines.
- `model/manifest.json`, `model/metadata.json` — model mappings and dataset metadata.
- `data/bank-additional-full.csv` — official public source data used in this run.

## Attribution

Dataset: UCI Machine Learning Repository, *Bank Marketing*, associated with work by Sérgio Moro, Paulo Cortez, and Paulo Rita. Refer to the UCI dataset page for its citation and license terms.