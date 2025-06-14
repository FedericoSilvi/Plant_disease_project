import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import mode


#dataset

# Model hyperparameter, taken from previous exp. Best five
model_params = {
    "XGBOOST_D3_1": {
        "model_class": XGBClassifier,
        "params": {
            "booster": "gblinear",
            "gamma": 0,
            "learning_rate": None,
            "max_depth": 3,
            "min_child_weight": 2,
            "n_estimators": 10
        }
    },
    "SVM_D3_1": {
        "model_class": SVC,
        "params": {
            "C": 100,
            "gamma": 0.001,
            "kernel": "rbf",
            "probability": True
        }
    },
    "SVM_D3_2": {
        "model_class": SVC,
        "params": {
            "C": 100,
            "gamma": 0.001,
            "kernel": "rbf",
            "probability": True
        }
    },
    "SVM_D4_2": {
        "model_class": SVC,
        "params": {
            "C": 100,
            "gamma": 0.001,
            "kernel": "rbf",
            "probability": True
        }
    },
    "XGBOOST_D4_2": {
        "model_class": XGBClassifier,
        "params": {
            "booster": "gblinear",
            "gamma": 0,
            "learning_rate": 1,
            "max_depth": 3,
            "min_child_weight": 2,
            "n_estimators": 50
        }
    }
}

# Set the parameters  based on the conf, about the normalization, feature selection and pca
def parse_pipeline_string(pipeline_str):
    # Parameters to set
    result = {
        "Normalization": None,
        "Feature_selection": None,
        "k" : None,
        "PCA": None,
        "n_components": None
    }
    if pipeline_str == "XGBOOST_D3_1":
        result["Normalization"] = "None"
        result["Feature_selection"] = "None"
        result["k"] = 0
        result["PCA"] = "Yes"
        result["n_components"] = 180

    elif pipeline_str == "SVM_D3_1":
        result["Normalization"] = "StandardScaler"
        result["Feature_selection"] = "Yes"
        result["k"] = 180
        result["PCA"] = "None"
        result["n_components"] = 0
    elif pipeline_str == "SVM_D4_2":
        result["Normalization"] = "StandardScaler"
        result["Feature_selection"] = "Yes"
        result["k"] = 160
        result["PCA"] = "None"
        result["n_components"] = 0
    elif pipeline_str == "SVM_D3_2":
        result["Normalization"] = "StandardScaler"
        result["Feature_selection"] = "Yes"
        result["k"] = 180
        result["PCA"] = "None"
        result["n_components"] = 0
    elif pipeline_str == "XGBOOST_D4_2":
        result["Normalization"] = "None"
        result["Feature_selection"] = "None"
        result["k"] = 0
        result["PCA"] = "Yes"
        result["n_components"] = 100
    return result


# Cross validation for each configuration
def run_crossval_and_save_predictions(conf_list):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for conf in conf_list:
        print(f"Running config: {conf}")

        # Recover the right configuration
        conf_diz = parse_pipeline_string(conf)

        # Load dataset based on config
        if conf == "XGBOOST_D3_1" or conf == "SVM_D3_1":
            df_all = pd.read_csv("LBP_max_GLCM_Alex(D3_1).csv")
        elif conf == "XGBOOST_D4_2" or conf == "SVM_D4_2":
            df_all = pd.read_csv("LBP_max_GLCM_Gabor_den(D4_2).csv")
        elif conf == "SVM_D3_2":
            df_all = pd.read_csv("LBP_max_GLCM_Alex_den(D3_2).csv")
        else:
            raise ValueError(f"Dataset not defined for configuration: {conf}")

        data = df_all.drop(columns=["label"])
        target_dataset = df_all["label"]
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target_dataset)
        feature_names = data.columns

        all_preds = np.zeros(len(target), dtype=object)

        for fold, (train_idx, test_idx) in enumerate(cv.split(data, target)):
            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]

            #Apply the conditions found the in the conf

            # Normalization
            if conf_diz["Normalization"] == "StandardScaler":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Feature selection
            if conf_diz["Feature_selection"] == "Yes":
                selector = SelectKBest(score_func=f_classif, k=conf_diz["k"])
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test)

            # PCA
            if conf_diz["PCA"] == "Yes":
                pca = PCA(n_components=conf_diz["n_components"])
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Model
            if conf.startswith("SVM"):
                model = SVC(**model_params[conf]["params"])
            elif conf.startswith("XGBOOST"):
                model = XGBClassifier(**model_params[conf]["params"])
            else:
                raise ValueError("Unsupported model type.")

            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)
            # Insert probabilities in the right place
            if isinstance(all_preds, np.ndarray) and all_preds.dtype == object:
                all_preds = np.zeros((len(target), y_proba.shape[1]))
            all_preds[test_idx] = y_proba


        # Save only the predictions
        df_pred = pd.DataFrame(all_preds, columns=[f"class_{i}" for i in range(all_preds.shape[1])])
        df_pred.to_csv(f"probas_{conf}.csv", index=False)
        print(f"Saved predictions to predictions_{conf}.csv")

# Best 5 config
conf_list = ["SVM_D3_1", "SVM_D3_2", "SVM_D4_2", "XGBOOST_D3_1", "XGBOOST_D4_2"]
# Run the crossval for all configuration
run_crossval_and_save_predictions(conf_list)


# Average choice: for every samples we average the predicted class probability and we choose the greatest

prob_dfs = []

for conf in conf_list:
    df = pd.read_csv(f"probas_{conf}.csv")
    prob_dfs.append(df)

# Probabilities sum
avg_proba = sum(prob_dfs) / len(prob_dfs)

# Take the class with the highest average probability
final_preds = np.argmax(avg_proba.values, axis=1)




# Load the dataset in order to take the true label
df_all = pd.read_csv("LBP_max_GLCM_Alex(D3_1).csv")
true_labels = df_all["label"]
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Final accuracy: ensamble predictions vs true labels
accuracy = accuracy_score(true_labels_encoded, final_preds)
print(f"Ensemble Soft Score Accuracy: {accuracy:.4f}")
