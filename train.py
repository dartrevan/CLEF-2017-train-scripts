from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import wordpunct_tokenize
from distances import cosine_similarities
from gensim.models import KeyedVectors
from argparse import ArgumentParser
from models import *
from utils import *
import pandas as pd

if __name__ == '__main__':
    argument_parser = ArgumentParser()
    argument_parser.add_argument('--embeddings', default=None)
    argument_parser.add_argument('--embeddings_dim', default=200)
    argument_parser.add_argument('--use_sim', action='store_true')
    argument_parser.add_argument('--dropout_rate', default=0.3, type=float)
    argument_parser.add_argument('--encoder_size', default=300, type=int)
    argument_parser.add_argument('--decoder_size', default=1000, type=int)
    argument_parser.add_argument('--data')
    args = argument_parser.parse_args()

    inputs = {}

    dataset = pd.read_csv(args.data)
    raw_texts, icd_codes = dataset.RawText, dataset.ICD10

    source_sequences = raw_texts.str.lower().apply(wordpunct_tokenize).values
    source_sequences, token_vocab = lookup(source_sequences)
    target_sequences = icd_codes.str.split(' ').values
    target_sequences, code_vocab = lookup(target_sequences)

    source_maxlen = max(map(len, source_sequences))
    target_maxlen = max(map(len, target_sequences))

    source_sequences = pad_sequences(
        source_sequences,
        source_maxlen,
        padding='post',
        truncating='post',
        value=token_vocab['<EOS>']
    )
    target_sequences = pad_sequences(
        target_sequences,
        target_maxlen,
        padding='post',
        truncating='post',
        value=0
    )
    target_sequences = np.expand_dims(target_sequences, 2)

    inputs['token_indexes'] = source_sequences

    similarity_size = None
    if args.use_sim:
        similarities = cosine_similarities(raw_texts)
        similarity_size = similarities.shape[1]
        inputs['similarities_in'] = similarities

    w2v_model = {}
    if args.embeddings is not None:
        w2v_model = KeyedVectors(args.embeddings, binary=True)
    weights = get_weights(w2v_model, token_vocab, args.embeddings_dim)

    seq2seq_model = seq2seq_bilstm(
        weights=weights,
        target_vocab_size=len(code_vocab),
        encoder_size=args.encoder_size,
        decoder_size=args.decoder_size,
        similarity_dropout_rate=args.dropout_rate,
        use_sim=args.use_sim,
        similarity_size=similarity_size,
        target_maxlen=target_maxlen,
        source_maxlen=source_maxlen
    )

    seq2seq_model.fit(inputs, target_sequences, epochs=10, batch_size=50)

    seq2seq_model.save('nn_models/seq2seq_nosim.bin')
    save_data('nn_models/seq2seq_token_index.bin', token_vocab)
    save_data('nn_models/seq2seq_icd_codes.bin', code_vocab)



