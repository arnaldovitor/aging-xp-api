import pandas as pd
from numpy import array


def split_sequence(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def separe_column(input_path, column):
    df = pd.read_csv(input_path, sep=" ")
    try:
        df = df.apply(lambda x: x.str.replace(',', '.'))
        data = pd.DataFrame(df)
        sequence = data[column].astype(float)
    except:
        data = pd.DataFrame(df)
        sequence = data[column]

    return sequence


def split_sets(sequence, train_perc):
    train_size = int(len(sequence) * train_perc)
    train, test = sequence[0:train_size], sequence[train_size:len(sequence)]
    return train, test


def normalize(sequence):
    s_min = sequence.min()
    s_max = sequence.max()
    sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return sequence, s_min, s_max


def inverse_normalize(sequence, s_min, s_max):
    sequence = sequence * (s_max - s_min) + s_min
    return sequence
