import pandas as pd
import numpy as np
import seaborn as sns
from typing import Literal

from copy import deepcopy
from pgmpy.models import BayesianModel
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import f1_score, classification_report, make_scorer
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def get_f1_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    verbose: Literal[0, 1, 2] = 0
) -> float:
    y_true = y_true.tolist()
    y_pred = y_pred.values.squeeze().tolist()
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    if verbose > 0:
        print("F1-score: %.2f" % f1)
    if verbose > 1:
        print(classification_report(y_true=y_true, y_pred=y_pred))
    return f1


def discretize(
    kbd: KBinsDiscretizer,
    df: pd.DataFrame
) -> pd.DataFrame:
    discrete_cols = df.select_dtypes(include='category').columns.values
    continuous_cols = [
        c for c in df.columns if c not in discrete_cols
    ]

    df_discretized = deepcopy(df)
    if len(continuous_cols) > 0:
        df_discretized[continuous_cols] = \
            kbd.transform(df[continuous_cols]).astype('int32')
    return df_discretized


def get_fixed_with_bn_dataset(return_kbd=False) -> pd.DataFrame:
    df = pd.read_csv("dataset/diabetes.csv").astype({
        "Pregnancies": "int32",
        "Glucose": "int32",
        "BloodPressure": "int32",
        "SkinThickness": "int32",
        "Insulin": "int32",
        "BMI": "int32",
        "DiabetesPedigreeFunction": "float",
        "Age": "int32",
        "Outcome": "category"
    })
    df.loc[:, 'Glucose':'BMI'] = df.loc[:, 'Glucose':'BMI'].replace(0, np.nan)
    coplete_rows = df.dropna().index

    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal').fit(
        df.iloc[coplete_rows].drop(columns="Outcome")
    )
    df.iloc[coplete_rows] = discretize(kbd, df.iloc[coplete_rows])
    bn = BayesianModel([
        ("Age", "Pregnancies"),
        ("BMI", "BloodPressure"),
        ("BMI", "Outcome"),
        ("Pregnancies", "Outcome"),
        ("DiabetesPedigreeFunction", "Outcome"),
        ("Outcome", "Glucose"),
        ("Outcome", "Insulin"),
        ("Insulin", "SkinThickness"),
    ])
    bn.fit(df.iloc[coplete_rows])

    to_fix = df[
        df["Glucose"].notna() & df["BloodPressure"].notna() &
        df["BMI"].notna() & df["Insulin"].isna() & df["SkinThickness"].isna()
    ]
    _fix_missing_values(df, to_fix, kbd, bn)
    to_fix = df[
        df["Glucose"].notna() & df["BloodPressure"].notna() &
        df["BMI"].notna() & df["Insulin"].isna() & df["SkinThickness"].notna()
    ]
    _fix_missing_values(df, to_fix, kbd, bn)
    df = df.dropna()
    if return_kbd:
        return df, kbd
    return df


def _fix_missing_values(
    df: pd.DataFrame,
    to_fix: pd.DataFrame,
    kbd: KBinsDiscretizer,
    bn: BayesianModel
) -> pd.DataFrame:
    if to_fix.shape[0] == 0:
        print("Warning: No records in \"to_fix\"")
        return
    missings = to_fix.columns[to_fix.isna().any()].tolist()
    to_fix = to_fix.fillna(0)
    to_fix = discretize(kbd, to_fix)
    to_fix.loc[to_fix.index, missings] = \
        bn.predict(to_fix.drop(columns=missings)).set_index(to_fix.index)
    df.loc[to_fix.index, :] = to_fix
    return df


def test_simple_classifiers(X, y):

    scorer = make_scorer(f1_score)

    results = {"Classifier": [], "f1_score": []}
    for Classifier in [
        GaussianNB, SVC, KNeighborsClassifier,
        DecisionTreeClassifier, AdaBoostClassifier, RandomForestClassifier
    ]:
        model = Classifier()
        results["Classifier"].append(str(model)[:-2])
        result = cross_val_score(
            model, X, y, cv=10, scoring=scorer
        )
        results["f1_score"].append(result)

    results = pd.DataFrame(data=results)
    results = results.explode(
        'f1_score'
    ).astype({"f1_score": "float"}).reset_index(drop=True)
    return results


def plot_simple_classifiers_results(results: pd.DataFrame):
    print(
        "Best model %s got f1-score %0.3f" %
        tuple(results.iloc[
            results["f1_score"].idxmax()
        ][["Classifier", "f1_score"]])
    )
    sns.catplot(
        x="f1_score",
        y="Classifier",
        hue="dataset",
        data=results
    )
