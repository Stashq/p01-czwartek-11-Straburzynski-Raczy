import pandas as pd
from typing import Literal

from copy import deepcopy
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import f1_score, classification_report


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

def fix_missing_values(
    kbd: KBinsDiscretizer,
    df: pd.DataFrame
) -> pd.DataFrame:
    
