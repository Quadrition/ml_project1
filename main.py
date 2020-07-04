import pandas as pd
from kmeans import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster
from sklearn import tree
import numpy as np
import copy
import seaborn

def normalize_data(data):
    cols = len(data[0])

    for col in range(cols):
        column_data = []
        for row in data:
            column_data.append(row[col])

        mean = np.mean(column_data)
        std = np.std(column_data)

        for row in data:
            row[col] = (row[col] - mean) / std


def analyze_with_pca():
    df = pd.read_csv('credit_card_data.csv')
    df = df.fillna(df.median())
    original_data = df.iloc[:, 1:].values
    data = copy.deepcopy(original_data)

    normalize_data(data)

    pca = PCA()
    pca.fit(data)

    plt.plot(range(1, 18), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.show()

    features = 7
    pca = PCA(n_components=features)
    pca.fit(data)
    scores = pca.transform(data)

    plt.bar(range(pca.n_components_), pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('Variance %')
    plt.xticks(range(pca.n_components_))
    plt.show()

    # plot_optimal_k(data)

    n_clusters = 9

    k_means = KMeans(n_clusters=n_clusters, max_iter=100)
    y_predict = k_means.fit(scores)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(original_data, y_predict)

    text_tree = tree.export_text(clf, feature_names=list(df.columns)[1:])
    print(text_tree)

    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'orange', 5: 'cyan', 6: 'yellow', 7: 'indigo', 8: 'pink',
              9: 'black'}
    plt.figure()
    for idx, cluster in enumerate(k_means.clusters):
        plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=100)
        for datum in cluster.data:
            plt.scatter(datum[0], datum[1], c=colors[idx])

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def analyze_without_pca():
    df = pd.read_csv('credit_card_data.csv')
    df = df.fillna(df.median())

    best_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']

    data = df[best_cols].iloc[:, 1:].values

    # plot_optimal_k(data)

    n_clusters = 8

    k_means = cluster.KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300)
    y = k_means.fit_predict(data)

    df['cluster'] = y
    best_cols.append('cluster')

    seaborn.pairplot(df[best_cols], hue='cluster')
    plt.show()


def plot_optimal_k(data):
    plt.figure()
    sum_squared_errors = []
    for n_clusters in range(1, 20):
        k_means = KMeans(n_clusters=n_clusters, max_iter=100)
        k_means.fit(data)
        sse = k_means.sum_squared_error()
        sum_squared_errors.append(sse)
    print(sum_squared_errors)
    plt.plot(range(1, 20), sum_squared_errors)
    plt.xlabel('# of clusters')
    plt.ylabel('WCSSE')
    plt.show()


def main():
    analyze_with_pca()
    # analyze_without_pca()


if __name__ == '__main__':
    main()
