'''
#Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
**Notes**
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
# from __future__ import print_function

from keras.preprocessing import sequence
from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding
from keras.layers import LSTM, GRU, Input, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional
from keras import optimizers
from random import shuffle
import numpy as np
import keras
import math
import random
import fileinput
import os
from bloom_filter import BloomFilter
import pickle
class gen(keras.utils.Sequence):
    def __init__(self, batch_size, data_path):
        self.batch_size = batch_size
        self.total = sum([len(files) for r, d, files in os.walk(data_path)])
        # self.total=len(os.listdir(data_path))
        self.data_path = data_path
        self.datas = os.listdir(data_path)

    def __len__(self):
        number_files = int(math.floor(self.total / self.batch_size))
        return number_files

    # def __getitem__(self, idx):
    #     x,y=[],[]
    #
    #     for i in range(self.batch_size):
    #         data = np.load(self.data_path+'train_{0}.npz'.format(idx * self.batch_size + i))
    #         x.append(data['name1'])
    #         y.append(data['name2'])
    #
    #     # data = np.load(self.data_path + 'train_{0}.npz'.format(idx * self.batch_size + idx))
    #     # x = data['name1']
    #     # y = data['name2']
    #     return np.asarray(x), np.asarray(y)
    def __getitem__(self, idx):
        indexes = random.sample(range(0, len(self.datas)), self.batch_size)
        files = [self.datas[i] for i in indexes]
        try:
            x, y = [], []
            for file_name in files:
                data = np.load(self.data_path + file_name)
                # print(self.data_path + file_name)
                x.append(data['name1'])
                y.append(data['name2'])
        except IndexError:
            indexes = random.sample(range(0, len(self.datas)), self.batch_size)
            files = [self.datas[i] for i in indexes]
            x, y = [], []
            for file_name in files:
                data = np.load(self.data_path + file_name)
                # print(self.data_path + file_name)
                x.append(data['name1'])
                y.append(data['name2'])

        return np.asarray(x), np.asarray(y)

def data_gen(n):
    counter = 0
    while counter < n:
        yield 0
        counter +=1

def one_hot_encoding(line):
    onehot = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    return np.asarray([onehot[i] for i in line])

def rev_one_hot_encoding(line):
    onehot = {0: 'A', 1: 'C', 2:'G',3 : 'T', 4: 'N'}
    return [onehot[i] for i in line]


def get_kmers(seq, k):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

def preprocess_to_npz(file_name, save_path, start, num):

    counter = 0
    for line in fileinput.input([file_name]):
        print(counter)
        if counter < start:
            counter+=1
            continue
        else:
            seq = one_hot_encoding(line[:-1])
            label = 0
        if counter >= start+num:
            break
        np.savez(save_path+'train_{0}.npz'.format(counter), name1=seq, name2=label)
        # if counter > 13439314: # 0.8 keys, 0.2 non-keys
        #     save_label.append(0)
        # else:
        #     save_label.append(1)
        counter += 1

    pass

def check(path):
    for counter in range(2711000-1, 2711000-10, -1):
        file_name = path+'train_{0}.npz'.format(counter)
        data = np.load(file_name)
        x, y = data['name1'], data['name2']
        print(x, y)
    pass

def build_overflow_bloom(model, directory, threshold, bloomsize, err=0.05):

    all_reads = os.listdir(directory)
    bloom = BloomFilter(max_elements=bloomsize, error_rate=err)
    for read in all_reads:
        data = np.load(directory + read)
        prob = model.predict(np.asarray([data['name1']]))
        # print(prob)
        if prob < threshold:
            bloom.add(''.join(rev_one_hot_encoding(data['name1'])))
    return bloom


def get_accuracy(model, bloomfilter, directory, threshold):
    all_reads = os.listdir(directory)
    metrics = [0, 0, 0, 0] # TP, FP, TN, FN
    for read in all_reads:
        data = np.load(directory + read)
        read_string = ''.join(rev_one_hot_encoding(data['name1']))
        # negatives
        predict = model.predict(np.asarray([data['name1']]))
        if  predict< threshold and read_string not in bloomfilter:
            if data['name2'] == 0: # TN
                metrics[2] += 1
            else:
                metrics[3] += 1
        elif predict > threshold:
            if data['name2'] == 1: # TP
                metrics[0] += 1
            else: # FP
                metrics[1] += 1
    TP, FP, TN, FN = metrics[0], metrics[1], metrics[2], metrics[3]
    if TN+FN == 0:
        FNR = 0
    else:
        FNR = FN/(TN+FN)
    accuracy = (TP+TN)/(TP+FP+TN+FN )
    print('The accuracy is {}, and the FNR is {}, {}, {}'.format(accuracy, FNR, FP, TP))
    return accuracy, FNR

if __name__ == '__main__':
    max_features = 4
    input_len = 200
    # cut texts after this number of words (among top max_features most common words)
    k = 20
    batch_size = 32
    # start 2710779
    # data = np.load('train.npz')
    print('Loading data...')
    # preprocess_to_npz('D:/University_Work/Spring 2019/566/final/train.txt', 'D:/University_Work/Spring 2019/566/train_data/', 2710779, 2950000-2710779)

    train_path = 'D:/University_Work/Spring 2019/566/train_data/'
    eval_path = 'D:/University_Work/Spring 2019/566/eval_data/'
    small_eval = 'D:/University_Work/Spring 2019/566/small_eval/'
    temp_path = 'D:/University_Work/Spring 2019/566/temp0/'
    total = 2711000
    # check('D:/University_Work/Spring 2019/566/final/train_data/')
    # a = np.asarray([[1,2],[3,4]])
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    # a = len(os.listdir(small_eval))
    train_gen = gen(batch_size, train_path)
    eval_gen = gen(batch_size, eval_path)
    test_gen = gen(6, temp_path)
    # (x_test, y_test) =
    #
    opt = optimizers.adagrad(lr=1e-3)
    #
    # print('Build model...')
    # model = Sequential()
    # # model.add(Conv1D(32, kernel_size=(20), activation='relu', input_shape=(200, 1)))
    # # model.add(MaxPooling1D(pool_size=(10), strides=(2)))
    # # model.add(Input(shape=(200, 1), name='read'))
    # model.add(Embedding(input_len, 32, input_length=200, input_shape=(input_len, )))
    # model.add(BatchNormalization())
    # model.add(Bidirectional(GRU(900, return_sequences=False)))
    # # model.add(Dense(256, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    #
    # # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy',
    #               optimizer=opt,
    #               metrics=['accuracy'])
    # plot_model(model, to_file='multilayer_perceptron_graph.png')
    # print(model.input_shape)
    # print(model.output_shape)
    # print(model.summary())
    #
    # print('Train...')
    model = load_model('preliminary7.h5')
    predictions = model.evaluate_generator(gen(64, eval_path ))
    print(predictions)
    #
    directory = small_eval
    threshold = 0.6

    bloomsize = len(os.listdir(small_eval)) * 0.4
    b_filter = build_overflow_bloom(model, directory, threshold, bloomsize, err=0.05)
    with open('testfilters.bloom', 'wb') as testfilter:

        # Step 3
        pickle.dump(b_filter, testfilter)

    with open('testfilters.bloom', 'rb') as b_filter_file:

        # Step 3
        b_filter = pickle.load(b_filter_file)
    threshold = 0.6
    get_accuracy(model, b_filter, directory, threshold)

    # tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=0, batch_size=batch_size,
    #                                           write_graph=True, write_grads=False,
    #                                           write_images=False, embeddings_freq=0,
    #                                           embeddings_layer_names=None,
    #                                           embeddings_metadata=None, embeddings_data=None)
    #
    # model.fit_generator(generator=eval_gen,
    #                     # validation_data=eval_gen,
    #                     shuffle=True,
    #                     epochs = 20,
    #                     # use_multiprocessing=True,
    #                     # workers=4,
    #                     callbacks=[tensorboard]
    #                     )





    # incorrects = np.nonzero(predictions.reshape((-1,)))
    # FN, FP, TP, TN = 0, 0, 0,0
    # for i in range(len(os.listdir(small_eval))):
    #     if predictions[i] > 0.5 and i > 500:
    #         # predicted key but actually non-key
    #         FP += 1
    #     elif predictions[i] > 0.5 and i <= 500:
    #         TP += 1
    #     elif predictions[i] < 0.5 and i <= 500:
    #         # predicted nokey but actually key
    #         FN += 1
    #     elif predictions[i] > 0.5 and i > 500:
    #         # predicted nokey but actually key
    #         TN += 1




    # print('FP = {}, {}, FN={}, {}'.format(FP, str(FP/(FP+TP)), FN, str(FN/(FN+TN))))

    model.save('preliminary7.h5')
    # score, acc = model.evaluate(x_test, y_test,
    #                             batch_size=batch_size)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    
