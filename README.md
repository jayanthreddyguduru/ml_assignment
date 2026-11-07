# Bank Marketing Classification — ML Assignment 2

> **Student customization required:** Replace the placeholders, execute on BITS Virtual Lab, inspect the metrics, write your own findings and capture proof of execution. This is a learning scaffold, not a finished submission.

## a. Problem statement
Predict whether a Portuguese bank customer will subscribe to a term deposit (`y=yes` or `no`) using customer, campaign and economic indicators. Compare five classification algorithms on a fixed hold-out test set.

## b. Dataset description
- Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Selected variant: `bank-additional-full.csv`; 41,188 rows; 20 input features and 1 binary target.
- Original dataset includes numerical and categorical features.
- The `duration` variable is *excluded* for pre-call prediction because call duration is unknown before calling.
- We use 19 of the 20 original input features.
- Stratified random 80/20 train/test split (`random_state=42`). The original CSV is time ordered; a chronological holdout would be a useful additional experiment.
- Categorical predictors are imputed and one-hot encoded. Numeric predictors are imputed and standardized. All transformations are fit *only on training data* inside pipelines.
- `unknown` categories in the original dataset are preserved rather than silently dropped.

## c. GitHub Repository Link
ADD YOUR REPOSITORY URL HERE

## d. Models used and comparison
The training script fits Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and RBF Support Vector Machine on the same split. It exports measured Accuracy, ROC AUC, Precision, Recall, F1 and MCC to `model_comparison.csv`.

**Do not invent results.** After running the script, copy actual values from `model_comparison.csv` here.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD | TBD | TBD |
| Naive Bayes | TBD | TBD | TBD | TBD | TBD | TBD |
| Support Vector Machine | TBD | TBD | TBD | TBD | TBD | TBD |

### Per-model performance observations (write after examining *your* outputs)
| Model | Observation about model performance |
|---|---|
| Logistic Regression | TBD: interpret precision/recall/AUC and linear assumptions |
| Decision Tree | TBD: discuss interpretability and generalization |
| KNN | TBD: discuss distance, dimensionality and performance |
| Naive Bayes | TBD: discuss independence assumption and outcomes |
| Support Vector Machine | TBD: discuss nonlinear decision boundary and results |
| Overall winner | TBD: justify with measured metrics, especially F1, MCC, and AUC because classes are imbalanced |

## How to run on BITS Virtual Lab
1. Copy this project to BITS Virtual Lab.
2. Create a virtual environment: `python -m venv .venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run `python train_models.py` (downloads the public UCI dataset if not already available; alternatively place the official CSV in `data/`). Training may take time, particularly SVM.
5. Verify generated `model/*.joblib`, `model/manifest.json`, `model/metadata.json`, `test_data.csv`, and `model_comparison.csv`.
6. Capture an authentic screenshot showing successful script execution in the **BITS Virtual Lab**.
7. Run `streamlit run app.py` and upload `test_data.csv`.
8. Review outputs and replace all TBD observations with your own interpretations.
9. Commit source, README, requirements, test_data.csv, comparison CSV and the five generated serialized models. Do not commit raw dataset ZIP if unnecessary.
10. Deploy `app.py` to Streamlit Community Cloud from your GitHub repo. Test the live app with the provided CSV.

## Files
- `train_models.py`: downloads the official UCI data, splits, trains five models, evaluates, and saves fitted preprocessing+classifier pipelines.
- `app.py`: loads the five saved pipelines and evaluates uploaded labelled test data **without retraining**.
- `requirements.txt`: project dependencies.
- `test_data.csv`: generated held-out test data, including target.
- `model/`: generated fitted models and metadata.
- `model_comparison.csv`: generated six-metric comparison.

## Limitations and ethical considerations
The dataset reflects historic phone campaigns and may not transfer to current campaigns. Original sample is strongly imbalanced; accuracy alone is not a reliable selection criterion. While stratification balances the holdout labels, a future-time test is additionally important for a real deployment. The UI evaluates labelled CSVs, not arbitrary unlabeled leads.

## Live Streamlit App Link
ADD YOUR DEPLOYED APP URL HERE

## Attribution
Dataset: Moro, Cortez & Rita, Bank Marketing, UCI Machine Learning Repository. Original dataset licence and citation terms apply.
