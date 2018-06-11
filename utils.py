import pandas as pd
import numpy as np
import codecs
import pickle


def lookup(sequences, vocab=None, update=True):
    if vocab is None:
        vocab = {'<EOS>': 0, '<UNKNOWN>': 1}
    mapped_sequences = []
    for sequence in sequences:
        mapped_sequence = []
        for token in sequence:
            if token not in vocab:
                if update:
                    vocab[token] = len(vocab)
                else:
                    token = '<UNKNOWN>'
            mapped_sequence.append(vocab[token])
        mapped_sequences.append(mapped_sequence)

    return mapped_sequences, vocab


def one_hot_encode(sequences, vocab_size):
    sequences_count = sequences.shape[0]
    sequence_lengths = sequences.shape[1]
    one_hots = np.zeros((sequences_count, sequence_lengths, vocab_size))
    for i, sequence in enumerate(sequences):
        for j, token_idx in enumerate(sequence):
            one_hots[i, j, token_idx] = 1.0
    return one_hots


def get_weights(w2v_model, vocab, emb_dim):
    weights = np.random.randn(len(vocab), emb_dim)
    for token, index in vocab.iteritems():
        if token in w2v_model: weights[index] = w2v_model[token]

    return weights


def save_data(path, data):
    with codecs.open(path, 'wb') as out_file:
        pickle.dump(data, out_file)


def toDF(index, sequences):
    df = []
    for row, seq in zip(index, sequences):
        ind = row
        for i, code in enumerate(seq):
            if code == '<EOS>': break
            df.append({
                'DocID': ind[0],
                'YearCoded': ind[1],
                'LineID': ind[2],
                'Rank': i + 1,
                'StandardText': None,
                'ICD10': code,
            })

    return pd.DataFrame(df)
