import math
from itertools import cycle
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributions as dist
from matplotlib import cm, patches, animation
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.manifold import TSNE
from tqdm.auto import trange

from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import is_tensor, as_tensor

def visualize_tsne(
    points: Optional[np.ndarray] = None,
    gaussians: Optional[List[Tuple[np.ndarray, np.ndarray, float]]] = None,
    fig: Optional[plt.Figure] = None,
    lims: int = 2,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """Wizualizacja mikstury rozkładów normalnych. Wizualizacja w przestrzeni po T-SNE
    
    :param points: punkty do wizualizacji
    :param gaussians: lista krotek zawierających
        macierz średnich, macierz kowariancji i mixing coefficients
    :param fig: figure do rysowania
    """
    pass

def visualize2d(
    points: Optional[np.ndarray] = None,
    gaussians: Optional[List[Tuple[np.ndarray, np.ndarray, float]]] = None,
    fig: Optional[plt.Figure] = None,
    lims: int = 2,
) -> Tuple[plt.Figure, Tuple[plt.Axes, ...]]:
    """Wizualizacja mikstury rozkładów normalnych.
    
    :param points: punkty 2D do wizualizacji
    :param gaussians: lista krotek zawierających
        macierz średnich, macierz kowariancji i mixing coefficients
    :param fig: figure do rysowania
    """
    handles = []
    labels = []
    if fig is None:
        fig = plt.figure(figsize=(14, 5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection="3d")
    else:
        ax1, ax2, ax3 = fig.get_axes()
        ax1.clear()
        ax2.clear()
        ax3.clear()
    
    ax1.set_xlim(-lims, lims)
    ax1.set_ylim(-lims, lims)
    ax2.set_xlim(-lims, lims)
    ax2.set_ylim(-lims, lims)
    ax3.set_xlim(-lims, lims)
    ax3.set_ylim(-lims, lims)
    ax3.set_zlim(0, 0.5)
        
    if points is not None:
        ax1.scatter(
            points[:, 0], 
            points[:, 1],
            s=4,
            c="black",
            alpha=0.3, 
            marker="x",
            label="punkty"
        )
        h, l = ax1.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    if gaussians is not None:
        cmap = cycle(CMAP(i) for i in range(len(gaussians)))
        x = np.linspace(-lims, lims, 201)
        y = np.linspace(-lims, lims, 201)
        xs, ys = np.meshgrid(x, y)
        positions = np.stack((xs, ys), axis=2)
        zs = np.zeros_like(xs)

        for idx, (mu, sigma, pi) in enumerate(gaussians):
            c = next(cmap)
            label = f"komponent {idx}"
            rv = multivariate_normal(mu, sigma)
            zk = rv.pdf(positions)
            zs += pi * zk
            contour = ax1.contour(xs, ys, zk, levels=3, colors=[c])
            handles.append(patches.Patch(color=c, label=label))
            labels.append(label)

        ax2.contour(xs, ys, zs, levels=5)

        ax3.plot_surface(xs, ys, zs, cmap=cm.coolwarm)
        
    fig.legend(handles, labels, loc='center left')
    return fig, (ax1, ax2, ax3)

class GaussianMixtureModel:
    distribution = dist.MultivariateNormal
    def __init__(
        self,
        n_components: int, 
        n_dims: int,
        n_epochs: int = 50,
        termination_threshold: float = 1e-3, 
        verbose: bool = True
    ):
        """Inicjalizacja modelu dwuwymiarowego.
        
        :param n_components: liczba komponentów modelu
        :param n_epochs: liczba epok trenowania modelu
        :param termination_threshold: próg przyrostu log likelihood do zatrzymywania
        :param verbose: wyświetl informację o trenowaniu
        """
        self.history = {
            "means": [],
            "covariances": [],
            "mixing_coefs": [],
            "log_likelihood": []
        }
        
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.dim = n_dims
        self.termination_threshold = termination_threshold
        self.verbose = verbose
        
        self.means = torch.rand((self.n_components, self.dim))
        self.covariances = torch.eye(self.dim).repeat(
            self.n_components, 1, 1
        )
        self.mixing_coefs = torch.full(
            (self.n_components,),
            fill_value=1 / self.n_components
        )
        
    def expectation_step(self, x: torch.Tensor) -> torch.Tensor:
        """Krok Expectation (obliczenie responsibilities)."""
        # oblicz responsibilities zgodnie z równaniem
        resp = torch.tensor([
            [self.mixing_coefs[i]] *\
            multivariate_normal.pdf(x, self.means[i], self.covariances[i], allow_singular=True)
            for i in range(self.n_components)
        ], dtype=torch.float32)
        resp = resp/resp.sum(axis=0, keepdims=True)
        return resp.T
    
    def maximization_step(
        self, x: torch.Tensor, resp: torch.Tensor
    ):
        """Krok Maximization (obliczenie parametrów)."""
        # zaktualizuj parametry modelu zgodnie z równaniami
        N_k = resp.sum(axis=0)
        mixing_coefs = N_k / torch.sum(N_k)
        
        covariances = []
        for k in range(self.n_components):
            r = resp[:, k]
            var = x - self.means[k]
            # sumowanie jest już zawarte w operacji na wszystkich próbkach
            covariances.append(r * var.T @ var / N_k[k])
        covariances = torch.stack(covariances)
        
        resp = torch.unsqueeze(resp, dim=2)
        x    = torch.unsqueeze(x,    dim=1)
        means = torch.sum(resp * x,  dim=0) / torch.unsqueeze(N_k, dim=-1)
        
        self.means = means
        self.covariances = covariances
        self.mixing_coefs = mixing_coefs
        
    def loglikelihood(self, x: torch.Tensor) -> float:
        """Log-likelihood modelu."""
        # oblicz średni log likelihood zgodnie z równaniem
        likelihood = torch.tensor([
            [self.mixing_coefs[i]] *\
            multivariate_normal.pdf(x, self.means[i], self.covariances[i], allow_singular=True)
            for i in range(self.n_components)
        ])
        likelihood  = torch.sum(likelihood, dim=0)
        likelihood  = torch.log(likelihood)
        avg_logprob = torch.sum(likelihood)
        return avg_logprob

    def update_history(self, x: torch.Tensor):
        """Aktualizacja logu."""
        self.history["means"].append(self.means)
        self.history["covariances"].append(self.covariances)
        self.history["mixing_coefs"].append(self.mixing_coefs)
        self.history["log_likelihood"].append(self.loglikelihood(x))
    
    def fit(self, x: Union[torch.Tensor, np.ndarray]) -> float:
        """Trenowanie modelu."""
        if not is_tensor(x):
            x = as_tensor(x)

        self.update_history(x)
        for epoch in trange(self.n_epochs):
            responsibilities = self.expectation_step(x)
            self.maximization_step(x, responsibilities)
            self.update_history(x)
            if (
                self.history["log_likelihood"][-1] - self.history["log_likelihood"][-2] 
                < self.termination_threshold
            ):
                print("Log likelihood not increasing, stopping.")
                self.n_epochs = epoch
                break
        
        if self.verbose:
            self._plot_training()
        
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Predykcja posterior dla każdego z komponentów."""
        if not is_tensor(x):
            x = as_tensor(x)

        return self.expectation_step(x)
    
    def pdf(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not is_tensor(x):
            x = as_tensor(x)
            
        resp = torch.tensor([
            [self.mixing_coefs[i]] *\
            multivariate_normal.pdf(x, self.means[i], self.covariances[i], allow_singular=True)
            for i in range(self.n_components)
        ], dtype=torch.float32)
        return resp.sum(axis=0)
        
    def _plot_training(self):
        """Wizualizacja uczenia."""
        fig, ax = plt.subplots()
        ax.plot(self.history["log_likelihood"])
        ax.set_xlabel("Epoka")
        ax.set_ylabel("Log likelihood")
        plt.show()
        
    def animation(self, x: torch.Tensor):
        """Animacja uczenia modelu."""
        fig, axes = visualize(points=x.numpy())
        anim = animation.FuncAnimation(
            fig=fig,
            func=lambda i: visualize(
                points=x.numpy(),
                gaussians=[
                    (mu, sigma, pi)
                    for mu, sigma, pi 
                    in zip(
                        self.history["means"][i].numpy(),
                        self.history["covariances"][i].numpy(),
                        self.history["mixing_coefs"][i].numpy(),
                    )
                ],
                fig=fig
            ), 
            frames=range(0, self.n_epochs, 5 if self.n_epochs > 10 else 1), 
            interval=500,
            blit=False
        )
        plt.close()
        return anim.to_jshtml()
    
    def sample(
        self, sample_shape: torch.Size = torch.Size([])
    ) -> torch.Tensor:
        """Próbkuj z modelu mikstur."""
        component = dist.Categorical(probs=self.mixing_coefs).sample()
        
        component_dist = dist.MultivariateNormal(self.means[component], self.covariances[component])
        return component_dist.sample(sample_shape=sample_shape)

    def aic(self, x: torch.Tensor) -> float:
        """Oblicz wartość AIC dla danych i parametrów modelu."""
        n_param_m = np.prod(list(self.means.shape))
        n_param_cov = np.prod(list(self.covariances.shape))
        n_param_coef = np.prod(list(self.mixing_coefs.shape))
        n_parameters = n_param_m + n_param_cov + n_param_coef
        
        return float(self.loglikelihood(x) - n_parameters)


    def bic(self, x: torch.Tensor) -> float:
        """Oblicz wartość BIC dla danych i parametru modelu."""
        n_param_m = np.prod(list(self.means.shape))
        n_param_cov = np.prod(list(self.covariances.shape))
        n_param_coef = np.prod(list(self.mixing_coefs.shape))
        n_parameters = n_param_m + n_param_cov + n_param_coef
        
        return float(self.loglikelihood(x) - 0.5 * n_parameters * math.log(len(x)))


class GaussianMixtureModelKMeans(GaussianMixtureModel):
    """GMM z inicjalizacją średnich z użyciem KMeans."""
    def fit(self, x: torch.Tensor, random_state = None) -> float:
        kmeans = KMeans(n_clusters=self.n_components, random_state=random_state)
        kmeans.fit(x)
        
        self.means = torch.tensor(kmeans.cluster_centers_, dtype=self.means.dtype)
        super(GaussianMixtureModelKMeans, self).fit(x)