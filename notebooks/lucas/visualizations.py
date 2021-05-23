import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import SelectMajorCategories


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


def plot_estimators_cvperf(estimators_list, figsize=(8, 4)):
    data = [pd.DataFrame(estimator.cv_results_)['mean_test_score'].dropna().values for estimator in estimators_list] 
    estimator_names = [str(estimator.estimator).replace('()', '') for estimator in estimators_list]
        
    plt.figure(figsize=figsize)
    plt.title('Model(s) CV performance')
    plt.boxplot(data)
    plt.xticks([i+1 for i in range(len(estimator_names))], estimator_names)
    plt.tight_layout()
    plt.show()
