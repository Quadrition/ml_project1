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


def get_description(cluster_index):
    if cluster_index == 0:
        return 'Korisnici najcesce menjaju balans od svih. Obicno ne placa puno novca unapred, ne radi to\n' \
               'cesto i nema puno transakcija sa uplacenim novcem unapred. Takodje, najredje od svih kupuju jednokratno.\n' \
               'Najduze im traju usluge kreditne kartice'
    elif cluster_index == 1:
        return 'Korisnicima je ukupna potrosnja najmanja od svih, a pojedinacna potrosnja im je takodje mala. Iznos\n' \
               'potrosen na kupovinu na rate im je najmanji od svih, imaju mali kreditni limit i usluge na krtici im\n' \
               'dugo vaze.'
    elif cluster_index == 2:
        return 'Korisnici imaju najveci balans od svih. Kreditni limit im je jedan od najvecih, a usluge kreditne\n' \
               'kartice im traju osrednje.'
    elif cluster_index == 3:
        return 'Korisnici najcesce imaju jednokratne kupovine. Uglavnom ne uplacuju puno unapred para unapred, ne rade\n' \
               'to cesto i nemaju mnogo transakcija vezanih za to. Minimalan iznos koji su uplatili na karticu im je mali\n' \
               'i dugo im vaze usluge kartice.'
    elif cluster_index == 4:
        return 'Korisnici imaju jedan od najmanjih balansa. Najredje od svih vrse kupovinu, i usluge kreditne kartice\n' \
               'im traju manje od svih.'
    elif cluster_index == 5:
        return 'Korisnici daju najmanje novca unapred, a i nemaju mnogo transakcija vezanih za to. Najcesce vrse kupovinu, \n' \
               'od svih, ali cesto i kupuju na rate. Ne uplacuju puno para unapred i nemaju puno transakcija vezanih za to.'
    elif cluster_index == 6:
        return 'Korisnici imaju najveci potroseni iznos od svih. Najvise trose na jednokratnu kupovinu i imaju najveci\n' \
               'iznos potrosen na kupovinu na rate. Cesto vrse kupovinu, imaju veliki limit na kredit i imaju veliki\n' \
               'broj transakcija vezanih za kupovinu.'
    elif cluster_index == 7:
        return 'Korisnici cesto menjaju balans, daju najvise novca unapred, ali rade to malo cesce od ostalih i imaju\n' \
               'veci broj transakcija vezanih za to. Imaju drugo po redu najkrace vazenje usluga kreditne kartice.'
    else:
        return 'Korisnici imaju najmanji balans od svih, najredje im se menja balas i imaju najmanju potrosnju na\n' \
               'jednokratnu kupovinu.'


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

    columns = list(df.columns)[1:]

    summary = [[], [], [], [], [], [], [], [], []]
    for i in range(len(original_data)):
        summary[y_predict[i]].append(original_data[i])

    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'orange', 5: 'cyan', 6: 'yellow', 7: 'indigo', 8: 'pink',
              9: 'black'}

    print(
       '===================================================Summary===================================================')
    for i in range(len(summary)):
        if i != 0:
            print('\n\n')
        print('\nCluster ' + str(i + 1) + ', color: ' + colors[i])
        print('--------------------------------------------------------------------')
        print(get_description(i))
        print('--------------------------------------------------------------------')
        for j in range(len(columns)):
            print('Attribute ' + columns[j] + ':')
            print('\tMaximum value: ' + str(max([datum[j] for datum in summary[i]])))
            print('\tThird quartile: ' + str(np.percentile([datum[j] for datum in summary[i]], 75)))
            print('\tMean: ' + str(np.percentile([datum[j] for datum in summary[i]], 50)))
            print('\tFirst quartile: ' + str(np.percentile([datum[j] for datum in summary[i]], 25)))
            print('\tMinimum value: ' + str(min([datum[j] for datum in summary[i]])))

    print(
       '\n\n\n\n==========================================Decision tree==========================================\n\n')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(original_data, y_predict)

    text_tree = tree.export_text(clf, feature_names=list(columns))
    print(text_tree)

    plt.figure()
    for idx, cluster in enumerate(k_means.clusters):
        plt.scatter(cluster.center[0], cluster.center[1], c=colors[idx], marker='x', s=100)
        for datum in cluster.data:
            plt.scatter(datum[0], datum[1], c=colors[idx])

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

    print('Na sledecim graficima mozemo da primetimo sledece klastere')
    print('-----------------------------------------------------------')
    print('1. Grupa koja ima skupe kupovine i ima limit za kredit koji je nesto jaci od proseka. Ovo je mala grupa ljudi.')
    print('2. Grupa koja uglavnom uplacuje pare u napred i ima velike kupovine, koja je takodje mala grupa.')
    print('3. Grupa koja je druga po redu po skupim kupovinama (posle 1. grupe).')
    print('4. Grupa koja ima najveci limit za kredit ali je stedljiva (nema puno kupovina). Ovo je treca najveca grupa.')
    print('5. Grupa koja uglavnom uzima pare unapred, ali ima jeftine kupovine.')
    print('6. Grupa koja malo trosi i ima mali limit za kredit, imaju 2. po redu najmanji balans. Ovo je druga najveca grupa.')
    print('7. Grupa sa najmanjim limitom za kredit koja ima najjeftinije kupovine. Ovo je najveca grupa od svih.')
    print('8. Grupa ima najvece minimalne kupovine, ali imaju 2. po redu najmanji limit za kredit.')


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
