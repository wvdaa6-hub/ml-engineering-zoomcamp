import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


# Final model settings, chosen in the notebook with 5-fold cross-validation
MAX_DEPTH = 15
MIN_SAMPLES_LEAF = 1
N_ESTIMATORS = 200
OUTPUT_FILE = "model.bin"


# Load the data and make column names lowercase
def load_data(path="data/water_potability.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    return df


# Split into train (60%), validation (20%) and test (20%), stratified on the target
def split_data(df, seed=1):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["potability"])
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed, stratify=df_full_train["potability"])
    return [d.reset_index(drop=True) for d in (df_full_train, df_train, df_val, df_test)]


# Add distance from neutral pH as an extra feature (potable water is closer to pH 7)
def add_features(df):
    df = df.copy()
    df["ph_dist"] = abs(df["ph"] - 7)
    return df


# Random forest pipeline: median imputation + many independent trees, averaged
def make_rf(max_depth=None, min_samples_leaf=1, n_estimators=200):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=1,
            n_jobs=-1,
        ),
    )


if __name__ == "__main__":
    # 1. Load, split and add features
    df = load_data()
    df_full_train, _, _, df_test = split_data(df)
    df_full_train = add_features(df_full_train)
    df_test = add_features(df_test)

    # 2. Separate features and target
    y_full_train = df_full_train.pop("potability").values
    y_test = df_test.pop("potability").values

    # 3. Train the final model on train + validation
    model = make_rf(max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF, n_estimators=N_ESTIMATORS)
    model.fit(df_full_train, y_full_train)

    # 4. Check the score on the test set
    y_pred = model.predict_proba(df_test)[:, 1]
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_pred):.3f}")

    # 5. Save the model (the pipeline includes the imputer, so missing values are handled)
    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump(model, f_out)
    print(f"Model saved to {OUTPUT_FILE}")