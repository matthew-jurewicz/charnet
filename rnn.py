import sys, os, io
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dropout, TimeDistributed, Dense
from keras.optimizers import RMSprop
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler

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
        x = np.reshape(one_hot[:-1], (-1,seq_len,vocab_len))
        #label is next sequence
        y = np.reshape(one_hot[1:], (-1,seq_len,vocab_len))
        np.savez_compressed(data_save_file, x=x, y=y)

    return x, y, char2idx

def load_model(nneurons, drop_rate, nlayers, input_shape):
    model = Sequential()
    #stateful LSTM better for text generation
    model.add(CuDNNLSTM(
        batch_input_shape=input_shape, 
        units=nneurons, 
        return_sequences=True, 
        stateful=True
    ))

    for i in range(nlayers - 1):
        if drop_rate > 0:
            model.add(Dropout(rate=drop_rate))
        model.add(CuDNNLSTM(
            units=nneurons, 
            return_sequences=True, 
            stateful=True
        ))

    model.add(Dropout(rate=drop_rate))
    vocab_len = input_shape[-1]
    model.add(TimeDistributed(Dense(vocab_len, activation='softmax')))

    return model

#manually reset states for stateful LSTM
class ResetStates(Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.reset_states()

#https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning
def softmax(vals, temperature):
    tmp = np.exp(np.log(vals) / temperature)
    return tmp / np.sum(tmp)

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
        nneurons = 256
        drop_rate = .5
        nlayers = 3
        model = load_model(nneurons, drop_rate, nlayers, (1,seq_len,vocab_len))

        learn_rate = 1e-3
        if os.path.exists(model_save_file):
            model.load_weights(model_save_file)
            if transfer_learn:
                #smaller learning rate for fine-tuning
                learn_rate = 1e-4

        model.compile(
            optimizer=RMSprop(lr=learn_rate), 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )

        train = not os.path.exists(model_save_file) or transfer_learn
        if train:
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            checkpoint = ModelCheckpoint(
                filepath='checkpoints/rnn-{val_acc:.3f}.h5', 
                save_best_only=True, 
                save_weights_only=True
            )

            step_decay = lambda epoch: learn_rate * (.9 ** epoch)
            scheduler = LearningRateScheduler(schedule=step_decay)

            nepochs = 50
            model.fit(
                x=x, 
                y=y, 
                batch_size=1, 
                epochs=nepochs, 
                verbose=2, 
                callbacks=[ResetStates(), checkpoint, scheduler], 
                validation_split=.2, 
                shuffle=False
            )

    idx2char = {v:k for k,v in char2idx.items()}
    #generate text
    n = 1000
    #start sequence
    x = x[0].tolist()
    s = ''
    for i in range(n):
        x2 = [[x]]
        #NOTE: update keras to use 64-bit floats
        pred = model.predict_on_batch(x2)[0][-1]
        pred = softmax(pred, .5)
        rand = np.random.choice(range(vocab_len), p=pred)
        s += idx2char[rand]

        one_hot = np.zeros(vocab_len)
        one_hot[rand] = 1
        x = x[1:] + [one_hot]
    print(s)