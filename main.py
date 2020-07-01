import pandas as pd
import numpy as np


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


def main():
    df = pd.read_csv('credit_card_data.csv')
    df = df.fillna(df.median())
    data = df.iloc[:, 1:].values

    normalize_data(data)
    # print(data)


if __name__ == '__main__':
    main()
