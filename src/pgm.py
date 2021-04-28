import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import BayesianModel
# requires graphviz and graphviz-dev installed in your operating system
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.layout import kamada_kawai_layout
from typing import Tuple


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
