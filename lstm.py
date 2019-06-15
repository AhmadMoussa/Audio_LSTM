# Audio LSTMv5

from __future__ import print_function
import tensorflow as tf
import numpy as np
import librosa

# Target log path
logs_path = '/tmp/tensorflow/audio-rnn'
writer = tf.summary.FileWriter(logs_path)

'''
    Audio Loader Module | Librosa + Tensorflow wrapper | Returns Tensor of shape (#of Audio Files, #of Samples, #of Channels)
    Load Audio | Preprocessing
'''
from audio_loader_and_encoder import lqe
quantization_channels = 256
loader = lqe()

# Returns Audio File given the path to said file
def load_audios(path, loader):
    audio = loader.load_audio(path, 8000, 1, True, False)
    audio_one_hot = loader.audio_to_one_hot(audio, 1, quantization_channels)
    print("Loaded Audio")
    min_audio = np.min(audio)
    max_audio = np.max(audio)

    # normalize
    audio = (audio - min_audio) / (max_audio - min_audio)
    print(audio.dtype, min_audio, max_audio)
    return audio, min_audio, max_audio
'''__________________________________________________________________________________________________________________________________________________________'''


'''
    Splits Audios into smaller slices to train on
'''
from tqdm import tqdm
def sample_generator(audio):
    # try to estimate next_sample (0 -255) based on 256 previous samples
    step = 5
    next_sample = []
    samples = []
    for j in tqdm(range(0, audio.shape[0] - maxlen, step)):
        seq = audio[j: j + maxlen + 1]
        seq_matrix = np.zeros((maxlen, nb_output), dtype = bool)
        for i,s in enumerate(seq):
            sample_ = int(s * (nb_output - 1)) # 0-255
            if i < maxlen:
                seq_matrix[i, sample_] = True
            else:
                seq_vec = np.zeros(nb_output, dtype=bool)
                seq_vec[sample_] = True
                next_sample.append(seq_vec)
                samples.append(seq_matrix)
    samples = np.array(samples, dtype=bool)
    next_sample = np.array(next_sample, dtype=bool)
    #print(samples.shape, next_sample.shape)
    return samples, next_sample
    '''
        Don't call fit in a loop, you're just retraining last part, read the issue here:
            https://stackoverflow.com/questions/51373088/is-there-a-difference-between-calling-fit-in-a-loop-vs-fit-with-batch-size/51373594
        Read the function descriptions here:
            https://keras.io/models/model/
    '''
'''__________________________________________________________________________________________________________________________________________________________'''

'''
    Data generator | Yields batches to train on
'''
directory_name = "Kick"
audio, min_audio, max_audio = 0, 0, 0
def audio_generator(list):
    for filename in list:
        print(directory_name + "\\" + str(filename))
        audio, min_audio, max_audio = load_audios(directory_name + "\\" + str(filename), loader)
        samples, next_sample = sample_generator(audio)
        yield samples, next_sample
'''__________________________________________________________________________________________________________________________________________________________'''

#This is just to organize the files on my pc
def list_sorter(audio_list):
    sorted = []
    for i, audio in enumerate(audio_list):
        sorted.append(str(i+1) + ".wav")
    return sorted

def generator_test():
    for generated in audio_generator(list_sorter(os.listdir(directory_name))):
        print("Generating", generated)
        input("Press Enter to continue...")

'''
    Parameters | Model Setup
'''
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import RMSprop

# so try to estimate next sample afte given (maxlen) samples
maxlen = 256 # 128 / sr = 0.016 sec
nb_output = 256  # resolution - 8bit encoding
latent_dim = 256

inputs = Input(shape=(maxlen, nb_output))
x = LSTM(latent_dim, return_sequences=True)(inputs)
x = Dropout(0.4)(x)
x = LSTM(latent_dim)(x)
x = Dropout(0.4)(x)
output = Dense(nb_output, activation='softmax')(x)
model = Model(inputs, output)

#optimizer = Adam(lr=0.005)
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
csv_logger = CSVLogger('training_audio.log')
escb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
# had error here https://keras.io/callbacks/
filepath = "models/audio-{epoch:.1f}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, period=2)

'''
    Helpers and Plotters
'''
# Plot loss function
import matplotlib.pyplot as plt
def plot_history():
    print("Training history")
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(model.history.history['loss'])
    ax1.set_title('loss')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(model.history.history['val_loss'])
    ax2.set_title('validation loss')
    plt.show()

# load array to audio buffer and play!!
from IPython.display import Audio, display

# predict and plot audio waveform
def test_model():
    def sample(preds, temperature=1.0, min_value=0, max_value=1):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        v = np.argmax(probas)/float(probas.shape[1])
        return v * (max_value - min_value) + min_value

    for start in range(5000,15000,10000):
        seq = audio[start: maxlen]
        seq_matrix = np.zeros((maxlen, nb_output), dtype=bool)
        for i,s in enumerate(seq):
            sample_ = int(s * (nb_output - 1)) # 0-255
            seq_matrix[i, sample_] = True

        for i in tqdm(range(5000)):
            z = model.predict(seq_matrix.reshape((1,maxlen,nb_output)))
            s = sample(z[0], 1.0)
            seq = np.append(seq, s)

            sample_ = int(s * (nb_output - 1))
            seq_vec = np.zeros(nb_output, dtype=bool)
            seq_vec[sample_] = True

            seq_matrix = np.vstack((seq_matrix, seq_vec))  # added generated note info
            seq_matrix = seq_matrix[1:]

        # scale back
        seq = seq * (max_audio - min_audio) + min_audio
        print("Saving audio")
        librosa.output.write_wav("audio_sample.wav", seq, 16000)
        # plot
        plt.figure(figsize=(30,5))
        plt.plot(seq.transpose())
        plt.show()

        display(Audio(seq, rate=16000))
'''__________________________________________________________________________________________________________________________________________________________'''

# run model
def run():
    audio_list = list_sorter(os.listdir(directory_name))
    model.fit_generator(audio_generator(audio_list), initial_epoch = 0, max_q_size = 2, steps_per_epoch = 64, shuffle = True, verbose = 1, epochs = 20, callbacks=[csv_logger, escb, checkpoint])
    plot_history()
    test_model()

run()
'''__________________________________________________________________________________________________________________________________________________________'''
