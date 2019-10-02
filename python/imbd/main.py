# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:11:39 2019

@author: Administrator
"""

#from keras.datasets import imdb
#
#(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=1000, skip_top=20, maxlen=None, start_char=1, oov_char=2, index_from=3)

from keras.datasets import imdb
NUM_WORDS=1000 # only use top 1000 words
INDEX_FROM=3   # word index offset

train,test = imdb.load_data(num_words=NUM_WORDS, index_from=INDEX_FROM)
train_x,train_y = train
test_x,test_y = test

word_to_id = imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in train_x[0] ))
