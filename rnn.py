import sys, os, io
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import RMSprop

def load_data(data_save_file, vocab_save_file, transfer_learn, seq_len):
    if os.path.exists(vocab_save_file):
        with np.load(vocab_save_file) as f:
            char2idx = f['char2idx'].tolist()
            vocab_len = len(char2idx)
    else:
        filelist = [os.path.join('data', filename) for filename in os.listdir('data')]
        #transfer learning data must share vocab
        filelist += [os.path.join('transfer_learning', filename) for filename in os.listdir('transfer_learning')]

        text = ''
        for filepath in filelist:
            with io.open(filepath, mode='r', encoding='utf8') as f:
                text += f.read()

        vocab = list(set(text))
        vocab_len = len(vocab)
        char2idx = {vocab[i]:i for i in range(vocab_len)}
        np.savez_compressed(vocab_save_file, char2idx=char2idx)

    if os.path.exists(data_save_file):
        with np.load(data_save_file) as f:
            x = f['x']
            y = f['y']
    else:
        data_dir = 'transfer_learning' if transfer_learn else 'data'
        text = ''
        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            with io.open(filepath, mode='r', encoding='utf8') as f:
                text += f.read() + '\n\n'

        idxs = [char2idx[ch] for ch in text]
        one_hot = to_categorical(idxs, num_classes=vocab_len).tolist()
        #padding
        pad_size = seq_len - ((len(one_hot) - 1) % seq_len)
        one_hot += [[0] * vocab_len] * pad_size

        #no label for last sequence
        x = np.reshape(one_hot[:-1], (-1,1,seq_len,vocab_len))
        #label is next sequence
        y = np.reshape(one_hot[1:], (-1,1,seq_len,vocab_len))
        np.savez_compressed(data_save_file, x=x, y=y)

    return x, y, char2idx

def load_model(nneurons, drop_rate, nlayers, input_shape):
    model = Sequential()
    #stateful LSTM better for text generation
    model.add(LSTM(
        batch_input_shape=input_shape, 
        units=nneurons, 
        dropout=drop_rate, 
        recurrent_dropout=drop_rate, 
        return_sequences=True, 
        stateful=True, 
        unroll=True
    ))

    for i in range(nlayers):
        model.add(LSTM(
            units=nneurons, 
            dropout=drop_rate, 
            recurrent_dropout=drop_rate, 
            return_sequences=True, 
            stateful=True, 
            unroll=True
        ))
    vocab_len = input_shape[-1]
    model.add(TimeDistributed(Dense(vocab_len, activation='sigmoid')))

    return model

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('python rnn.py [data save file] [vocab save file] [optional model save file] [optional transfer learning flag]')
    else:
        data_save_file, vocab_save_file = sys.argv[1:3]
        model_save_file = '' if len(sys.argv) == 3 else sys.argv[3]
        transfer_learn = len(sys.argv) == 5
        seq_len = 100
        x, y, char2idx = load_data(data_save_file, vocab_save_file, transfer_learn, seq_len)
        
        vocab_len = len(char2idx)
        nneurons = 128
        drop_rate = .5
        nlayers = 3
        model = load_model(nneurons, drop_rate, nlayers, (1,seq_len,vocab_len))

        learn_rate = 1e-3
        if transfer_learn:
            model.load_weights(model_save_file)
            #smaller learning rate for fine-tuning
            learn_rate = 1e-4

        model.compile(
            optimizer=RMSprop(lr=learn_rate), 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        pass