from keras.layers import LSTM, Input, RepeatVector, Dense, TimeDistributed, Dropout, GRU, GlobalAveragePooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.models import Model


def seq2seq_lstm(weights, target_vocab_size,
                 source_maxlen, target_maxlen):

    model_in = Input(shape=(source_maxlen,), dtype='int32')
    embeddings = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True)(model_in)

    encoder = LSTM(100, return_sequences=False)(embeddings)
    final_state = RepeatVector(target_maxlen)(encoder)
    decoder = LSTM(100, return_sequences=True)(final_state)
    out = TimeDistributed(Dense(target_vocab_size, activation='softmax'))(decoder)
    model = Model(inputs=model_in, outputs=out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def seq2seq_bilstm(weights, target_vocab_size, encoder_size,
                   decoder_size, similarity_dropout_rate, use_sim,
                   similarity_size, target_maxlen, source_maxlen):

    model_inputs = []
    tokens_in = Input(shape=(source_maxlen,), dtype='int32', name='token_indexes')
    model_inputs.append(tokens_in)
    embeddings = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights], trainable=True)(tokens_in)
    encoder = Bidirectional(GRU(encoder_size, return_sequences=False), merge_mode='concat')(embeddings)

    if use_sim:
        similarities_in = Input(shape=(similarity_size,), name='similarities_in')
        model_inputs.append(similarities_in)
        similarities_in_afert_dropout = Dropout(similarity_dropout_rate)(similarities_in)
        encoder = Concatenate(axis=-1)([encoder, similarities_in_afert_dropout])

    final_state = RepeatVector(target_maxlen)(encoder)
    decoder = GRU(decoder_size, return_sequences=True)(final_state)
    out = TimeDistributed(Dense(target_vocab_size, activation='softmax'))(decoder)
    model = Model(inputs=model_inputs, outputs=out)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model