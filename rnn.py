import sys, os, io
import numpy as np

from keras.utils import to_categorical

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

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('python rnn.py [data save file] [vocab save file] [optional model save file] [optional transfer learning flag]')
    else:
        data_save_file, vocab_save_file = sys.argv[1:3]
        model_save_file = '' if len(sys.argv) == 3 else sys.argv[3]
        transfer_learn = len(sys.argv) == 5
        seq_len = 100
        x, y, char2idx = load_data(data_save_file, vocab_save_file, transfer_learn, seq_len)
        pass