import numpy as np
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours, CondensedNearestNeighbour
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 1410
SKIP_BASIC = True
SKIP_ENN = True
SKIP_CNN = False


def generate_synthetic_data():
    results = []
    for i in range(5):
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=RANDOM_STATE + i,
        )
        results.append((X, y))

    for i in range(5):
        X, y = make_classification(
            n_samples=5000,
            n_features=10,
            n_classes=2,
            random_state=RANDOM_STATE + i + 5,
        )
        results.append((X, y))

    for i in range(5):
        X, y = make_classification(
            n_samples=10000,
            n_features=15,
            n_classes=2,
            random_state=RANDOM_STATE + i + 10,
        )
        results.append((X, y))
    return results


def read_scv(file_name):
    dataset = pd.read_csv(file_name)
    # dataset_desc = dataset.describe(include="all")

    dataset = dataset.dropna()  # drops column if at least 1 element is missing

    if "gender" in dataset:
        dataset["gender"] = dataset["gender"].replace({"Female": 0, "Male": 1, "female": 0, "male": 1})

    if file_name != "parkinsons.csv":
        X = dataset.iloc[:, :-1].to_numpy()
        y = dataset.iloc[:, dataset.shape[1] - 1].to_numpy()
    else:
        y = dataset['status'].to_numpy()
        X = dataset.drop(['name', 'status'], axis=1).iloc[:, :-1].to_numpy()
    return X, y


rskf = RepeatedStratifiedKFold(
    random_state=RANDOM_STATE
)
knn = KNeighborsClassifier(metric='manhattan')
K_BEST = 5
metric = balanced_accuracy_score
enn = EditedNearestNeighbours()
cnn = CondensedNearestNeighbour()

breast_data = load_breast_cancer(return_X_y=True)  # zbior Breast Cancer Wisconsin
X_breast = breast_data[0]
y_breast = breast_data[1]

diabetes = read_scv("diabetes.csv")
indian = read_scv("Indian Liver Patient Dataset (ILPD).csv")
mammographic = read_scv("mammo.csv")
parkinson = read_scv("parkinsons.csv")

datasets = [(X_breast, y_breast), diabetes, indian, mammographic,
            parkinson]  # (X_breast, y_breast), diabetes, indian, mammographic, parkinson

datasets.extend(generate_synthetic_data())

if not SKIP_BASIC:
    scores = [[] for _ in range(len(datasets))]
    for i, dataset in enumerate(datasets):
        X, y = dataset
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            f_classif_select = SelectKBest(k=K_BEST)
            X_best_train = f_classif_select.fit_transform(X_train, y_train)
            X_best_test = f_classif_select.fit_transform(X_test, y_test)

            knn.fit(X_best_train, y_train)
            y_pred = knn.predict(X_best_test)
            scores[i].append(metric(y_test, y_pred))

    for dataset_score in scores:
        print(np.mean(dataset_score))

if not SKIP_ENN:
    scores = [[] for _ in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        X, y = dataset
        X, y = enn.fit_resample(X, y)
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            f_classif_select = SelectKBest(k=K_BEST)
            X_best_train = f_classif_select.fit_transform(X_train, y_train)
            X_best_test = f_classif_select.fit_transform(X_test, y_test)

            knn.fit(X_best_train, y_train)
            y_pred = knn.predict(X_best_test)
            scores[i].append(metric(y_test, y_pred))

    for dataset_score in scores:
        print(np.mean(dataset_score))

if not SKIP_CNN:
    scores = [[] for _ in range(len(datasets))]

    for i, dataset in enumerate(datasets):
        X, y = dataset
        X, y = cnn.fit_resample(X, y)
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            f_classif_select = SelectKBest(k=K_BEST)
            X_best_train = f_classif_select.fit_transform(X_train, y_train)
            X_best_test = f_classif_select.fit_transform(X_test, y_test)

            knn.fit(X_best_train, y_train)
            y_pred = knn.predict(X_best_test)
            scores[i].append(metric(y_test, y_pred))

    for dataset_score in scores:
        print(np.mean(dataset_score))
