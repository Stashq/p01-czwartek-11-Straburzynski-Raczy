import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pgmpy.models import BayesianModel
# requires graphviz and graphviz-dev installed in your operating system
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.layout import kamada_kawai_layout
from typing import Tuple, Literal

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
    df: pd.DataFrame,
    n_bins: int = 5
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


def draw(
    bn: BayesianModel,
    figsize: Tuple[int, int] = None,
    layout: str = None,
    **kwargs
):
    plt.figure(figsize=figsize)
    if layout is None or layout == "graphviz_layout":
        pos = graphviz_layout(bn, prog='dot')
    if layout == "kamada_kawai_layout":
        pos = kamada_kawai_layout(bn, **kwargs)
    nx.draw(
        bn,
        pos=pos,
        with_labels=True,
        node_color='white',
        edgecolors='black',
        node_size=5000,
        arrowsize=10,
    )
    plt.xlim(
        min(list(map(lambda v: v[0], pos.values())))-15,
        max(list(map(lambda v: v[0], pos.values())))+15
    )
    plt.ylim(
        min(list(map(lambda v: v[1], pos.values())))-15,
        max(list(map(lambda v: v[1], pos.values())))+15
    )
    plt.show()
