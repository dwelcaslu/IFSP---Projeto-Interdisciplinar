import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from features.preprocessing import SelectMajorCategories


def plot_feature_boxplots(df_data, boxplot_cols, figsize=(12, 4), subplot_layout=(10, 1)):
    boxplot_cols = np.array(boxplot_cols)
    n_suplots = subplot_layout[0]*subplot_layout[1]
    n_boxplots = len(boxplot_cols)
    n_boxplots_per_suplot = int(n_boxplots/n_suplots) + 1 if n_boxplots%n_suplots > 0 else int(n_boxplots/n_suplots)
    plt.figure(figsize=figsize)
    for i in range(n_suplots):
        selected_cols = list(boxplot_cols[i*n_boxplots_per_suplot:(i+1)*n_boxplots_per_suplot])
        ax = plt.subplot(subplot_layout[0], subplot_layout[1], i+1)
        df_data.boxplot(column=selected_cols, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_feature_histograms(df_data, histogram_cols, major_cat_perc=0.01,  major_cat_dropna=True, figsize=(15, 3)):
    for i,col in enumerate(histogram_cols):
        plt.figure(figsize=figsize)
        try:
            dtype = df_data[col].dtype
            if dtype in ['float64', 'int64']:
                values = df_data[col].values
                values.sort()
                values = values.astype(str)
            else:
                values = df_data[col].values.astype(str)

            n_unique_values = len(np.unique(values))
            if n_unique_values > 200:
                sel = SelectMajorCategories(columns=[col], perc=major_cat_perc, dropna=major_cat_dropna)
                sel.fit(df_data)
                df_data = sel.transform(df_data)
                values = df_data[col].values.astype(str)
                n_unique_adjust = len(np.unique(values))
            else:
                n_unique_adjust = n_unique_values

            plt.title(f'{col} - ({n_unique_values}/{n_unique_adjust})')
            plt.hist(values, bins=n_unique_adjust)
            plt.xticks(rotation=-80)
        except KeyError:
            plt.title(col + ' (not found)')
        plt.show()


def plot_histograms_byclass(df_data, target_col, y_col, figsize=(8, 4),
                            bins_min=0, bins_max=100, n_bins=100, alpha=0.5):
    plt.figure(figsize=figsize)
    bins = np.linspace(bins_min, bins_max, n_bins)
    labels = df_data[target_col].unique()
    labels.sort()
    for x in labels:
        values = df_data[df_data[target_col]==x][y_col].values
        plt.hist(values, bins, alpha=alpha, label=x)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_scatter_byclass(df_data, target_col, cols, figsize=(8, 4), alpha=0.5):
    plt.figure(figsize=figsize)
    labels = df_data[target_col].unique()
    labels.sort()
    for x in labels:
        x_values = df_data[df_data[target_col]==x][cols[0]].values
        y_values = df_data[df_data[target_col]==x][cols[1]].values
        plt.scatter(x_values, y_values, alpha=alpha, label=x)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_embeddings(x_array, y_array, figsize=(12, 6), alpha=0.5):
    # Calculating the embeddings:
    pca = PCA(n_components=2)
    x_array_pca = pca.fit_transform(x_array)
    tsne = TSNE(n_components=2)
    x_array_tsne = tsne.fit_transform(x_array)
    
    
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.title(f'PCA - explained_variance_ratio: {round(sum(pca.explained_variance_ratio_), 4)}')
    for val in np.unique(y_array):
        x_values = x_array_pca[np.where(y_array == val)]
        plt.scatter(x_values[:, 0], x_values[:, 1], label=val, alpha=alpha)
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.title(f'TSNE - kl_divergence: {round(tsne.kl_divergence_, 4)}')
    for val in np.unique(y_array):
        x_values = x_array_tsne[np.where(y_array == val)]
        plt.scatter(x_values[:, 0], x_values[:, 1], label=val, alpha=alpha)
    plt.legend(loc='best')
    plt.show()
