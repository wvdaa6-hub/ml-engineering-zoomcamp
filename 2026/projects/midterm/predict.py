import pickle
import pandas as pd
from train import add_features


MODEL_FILE = "model.bin"

# The raw features the model expects, in the same order as during training
FEATURES = [
    "ph", "hardness", "solids", "chloramines", "sulfate",
    "conductivity", "organic_carbon", "trihalomethanes", "turbidity",
]


# Load the trained pipeline (imputer + random forest) from disk
def load_model(path=MODEL_FILE):
    with open(path, "rb") as f_in:
        return pickle.load(f_in)


# Predict the probability that one water sample is potable
def predict(model, sample):
    # One row, columns in training order; missing keys become NaN and are imputed by the pipeline
    df = pd.DataFrame([sample]).reindex(columns=FEATURES)
    df = add_features(df)

    probability = float(model.predict_proba(df)[0, 1])
    return {
        "potability_probability": round(probability, 3),
        "potable": probability >= 0.5,
    }


if __name__ == "__main__":
    # Quick local test with the first sample from the dataset
    model = load_model()
    sample = {
        "ph": 7.08, "hardness": 204.89, "solids": 20791.32, "chloramines": 7.30,
        "sulfate": 368.52, "conductivity": 564.31, "organic_carbon": 10.38,
        "trihalomethanes": 86.99, "turbidity": 2.96,
    }
    print(predict(model, sample))