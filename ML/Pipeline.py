from scipy.cluster.hierarchy import weighted
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

#It creates a pipeline with: normalization, feature selection and PCA, classifier.
def pipeline_creation(model_name):
    classifier = {
        "RF": RandomForestClassifier(),
        "SVM": SVC(),
        "XGBOOST": XGBClassifier(),
        "KNN": KNeighborsClassifier(),
    }.get(model_name, None)

    if classifier is None:
        raise ValueError(f"Unknown model name: {model_name}")

    return Pipeline([
        ('normalization', 'passthrough'),
        ('feature_selection', 'passthrough'),
        ('dim_reduction', 'passthrough'),
        ('classifier', classifier),
    ])
#Give us the number of k to try during the gridsearch for feature selection (with k best)
def parameters_span(dataset_name):
    par_span=[]
    if(dataset_name=="LBP_8_(D0_1).csv" or dataset_name=="LBP_8_den(D0_2).csv"):
        par_span=[5,10,15]
    elif(dataset_name=="LBP_8_12_16(D1_1).csv" or dataset_name=="LBP_8_12_16_den(D1_2).csv"):
        par_span = [10, 30, 50, 70]
    elif(dataset_name=="LBP_max_GLCM(D2_1).csv" or dataset_name=="LBP_max_GLCM(D2_2).csv"):
        par_span=[30,50,70,90]
    elif (dataset_name == "LBP_max_GLCM_Alex(D3_1).csv" or dataset_name == "LBP_max_GLCM_Alex_den(D3_2).csv"
          or dataset_name == "LBP_max_GLCM_Gabor(D4_1).csv" or dataset_name == "LBP_max_GLCM_Gabor_den(D4_2).csv"):
        par_span = [80, 100, 120, 140, 160, 180]
    return par_span

#Parameters of the pipeline are split in the common base than add the classifier__params
#it creates a list of possible combination of the pipeline
#in the end it will return a list with all possible combinations between type of normalization, feature selection, pca and hyp.
#the result of this function will be used for the gridsearch
def base_grid_blocks(dataset_name, classifier, classifier_params):
    span = parameters_span(dataset_name)
    # It's mandatory the normalization with SVM model
    if isinstance(classifier, SVC):
        scalers = [StandardScaler(), MinMaxScaler()]
    else:
        scalers = [StandardScaler(), MinMaxScaler(), 'passthrough']
    blocks = []

    # Feature selection only
    blocks.append({
        'normalization': scalers,
        'feature_selection': [SelectKBest(f_classif)],
        'feature_selection__k': span,
        'dim_reduction': ['passthrough'],
        'classifier': [classifier],
        **classifier_params
    })

    # PCA only
    blocks.append({
        'normalization': scalers,
        'feature_selection': ['passthrough'],
        'dim_reduction': [PCA()],
        'dim_reduction__n_components': span,
        'classifier': [classifier],
        **classifier_params
    })

    # No FS or PCA
    blocks.append({
        'normalization': scalers,
        'feature_selection': ['passthrough'],
        'dim_reduction': ['passthrough'],
        'classifier': [classifier],
        **classifier_params
    })
    return blocks
#build the true gridsearch: common base + hyperparameter of the classifier

def param_grid_creation(dataset_name, model_name):
    if model_name == "RF":
        return base_grid_blocks(dataset_name, RandomForestClassifier(), {})

    elif model_name == "KNN":
        return base_grid_blocks(dataset_name, KNeighborsClassifier(), {
            'classifier__n_neighbors': [5, 10, 15, 20]
        })

    elif model_name == "XGBOOST":
        return base_grid_blocks(dataset_name, XGBClassifier(), {
            'classifier__n_estimators': [10, 50],
            'classifier__learning_rate': [None, 0.1, 1],
            'classifier__booster': ["gbtree", "gblinear"],
            'classifier__max_depth': [3, 6, 12],
            'classifier__min_child_weight': [0.5, 1, 2],
            'classifier__gamma': [0, 1, 10]
        })

    elif model_name == "SVM":
        kernels = {
            'linear': {'classifier__C': [0.01, 0.1, 1, 10, 100]},
            'rbf': {'classifier__C': [0.01, 0.1, 1, 10, 100], 'classifier__gamma': [1e-3, 1e-2, 0.1, 'scale', 'auto']},
            'poly': {'classifier__C': [0.1, 1], 'classifier__degree': [2, 3], 'classifier__gamma': ['scale', 'auto']}
        }

        all_grids = []
        for kernel, params in kernels.items():
            all_grids += base_grid_blocks(dataset_name, SVC(kernel=kernel), params)
        return all_grids

    return []


#MAIN
print("\nDataset Loading...\n")

dataset_name="LBP_8_12_16_den(D1_2).csv"
#TO DO KNN(DONE), RF, SVM, XGBOOST
model_name="SVM"
result_file="D1_2-SVM.txt"
#DATASET LOADING

df_all = pd.read_csv(dataset_name)
target_column = "label"


data = df_all.drop(columns=[target_column])

# Check for missing or infinite values (in case stop the execution)
if(data.isnull().sum().sum()>0 or np.isinf(data).sum().sum()>0):
    print("Missing values or infinite values found!")
    print("Missing values:", data.isnull().sum().sum(), " | Infinite values:", np.isinf(data).sum().sum())
    raise ValueError("Error in dataset values")

# Check for categorical features (in case stop the execution)
categorical_features=data.select_dtypes(exclude=[np.number])
if(len(categorical_features)>0):
    print("Categorical features found!"
    raise ValueError("Error in dataset type")

target = df_all[target_column]
#encoding (no categorical feature, maybe just add a check for the prof). Maybe is needed encoding of the target
label_encoder = LabelEncoder()
target = pd.Series(label_encoder.fit_transform(target))

print("Dataset has been loaded correctly\n")

#PIPELINE CREATION

print("Pipeline creating...\n")

pipeline = pipeline_creation(model_name)
print("Pipeline created correctly\n")

print("Grid Search parameters creating...\n")

#GRID SEARCH PARAMETERS

#create the actual pipeline
param_grid = param_grid_creation(dataset_name,model_name)

print("Grid Search parameters created correctly\n")

scorer = make_scorer(accuracy_score)

best_params_list = []  # gather grid_search.best_params_ from each fold

#implementation of a nested cross validation in order to be more robust statistically (outer cv) and to find the best parameters
#for the experiment with a validation test (inner cv, with gridsearch)

# Outer CV: 5 folds
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_accuracies = []
test_precision=[]
test_recall=[]
test_f1score=[]


all_y_true = []
all_y_pred = []
fold_conf_matrices = []
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(data, target), 1):
    print(f"\n===== Outer Fold {fold_idx} =====\n")

    #split => _train is for tuning, _test for the final evaluation
    X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

    # Inner CV inside GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # inner 5-fold CV for hyperparameter tuning
        scoring=scorer,
        verbose=1,
        n_jobs=-1
    )

    print("Grid Search object created successfully\n")

    # This method will take the X_train, splits it in 5 folds, use 4 of them for training and one for validation.

    grid_search.fit(X_train, y_train)


    best_params_list.append(grid_search.best_params_)
    print("Best parameters: ", grid_search.best_params_)
    print("Best CV score (inner): ", grid_search.best_score_)

    # We save the best configuration obtained during the grid search
    best_configuration = grid_search.best_estimator_

    # Extract SVM kernel manually (if SVM is used)
    best_kernel = None
    if model_name == "SVM":
        best_kernel = best_configuration.named_steps['classifier'].kernel

    # Evaluate on the outer test fold, on X_test never seen before
    y_pred = best_configuration.predict(X_test)

    # In order to compute the confusion matrix for each fold and for the aggregated version
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    cm_fold = confusion_matrix(y_test, y_pred)
    fold_conf_matrices.append(cm_fold)

    # Score metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1=f1_score(y_test,y_pred,average='weighted')
    precision= precision_score(y_test,y_pred,average='weighted')
    recall=recall_score(y_test,y_pred,average='weighted')

    print(f"Test accuracy on outer fold {fold_idx}: {accuracy:.4f}")
    print(f"Test f1 score on outer fold {fold_idx}: {f1:.4f}")
    print(f"Test precision on outer fold {fold_idx}: {precision:.4f}")
    print(f"Test recall on outer fold {fold_idx}: {recall:.4f}")
    test_accuracies.append(accuracy)
    test_f1score.append(f1)
    test_recall.append(recall)
    test_precision.append(precision)


cm_agg = confusion_matrix(all_y_true, all_y_pred)

mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)
mean_f1=np.mean(test_f1score)
std_f1=np.std(test_f1score)
mean_precision=np.mean(test_precision)
std_precision=np.std(test_precision)
mean_recall=np.mean(test_recall)
std_recall=np.std(test_recall)

#AVERAGE PERFORMANCE THROUGH ALL EXPERIMENTS

print("\n===== Summary over Outer Folds =====")
print(f"Average Test Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation:    {std_accuracy:.4f}")
from collections import Counter

#Best parameters print (majority voting)

most_common_params = Counter(tuple(sorted(p.items())) for p in best_params_list).most_common(1)
print("Most common best parameters:\n")
for key, value in most_common_params:
        print(f"  {key}: {value}")

#Save results on file

with open(result_file, 'w') as f:
    f.write("=== MACHINE LEARNING CLASSIFICATION REPORT ===\n\n")
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Model: {model_name}\n")
    f.write("Best configuration:\n")
    for key, value in most_common_params:
        f.write(f"  {key}: {value}\n")
    if best_kernel:
        f.write(f"  classifier__kernel: '{best_kernel}'\n")
    f.write(f"Average accuracy: {mean_accuracy*100}%\n")
    f.write(f"Average deviation: {std_accuracy*100}%\n")
    f.write(f"Average f1 score: {mean_f1 * 100}%\n")
    f.write(f"Average f1 score: {std_f1 * 100}%\n")
    f.write(f"Average precision: {mean_precision * 100}%\n")
    f.write(f"Average precision: {std_precision * 100}%\n")
    f.write(f"Average recall: {mean_recall * 100}%\n")
    f.write(f"Average recall: {std_recall * 100}%\n")
    f.write("\nConfusion Matrices per Fold:\n")
    for i, cm in enumerate(fold_conf_matrices, 1):
        f.write(f"\nFold {i} Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n")

    f.write("\nAggregated Confusion Matrix:\n")
    f.write(np.array2string(cm_agg))