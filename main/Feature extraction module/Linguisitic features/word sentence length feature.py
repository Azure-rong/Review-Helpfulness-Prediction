#! /usr/bin/env python2.7
#coding=utf-8

"""
Counting review's word number, sentence number and review length
This module aim to extract review's word number and sentence number and review length features.

"""

import textprocessing as tp


# Function counting review word number, sentence number and review length
def word_sent_count(dataset):
    word_sent_count = []
    for review in dataset:
        sents = tp.cut_sentence_2(review)
        words = tp.segmentation(review,'list')
        sent_num = len(sents)
        word_num = len(words)
        sent_word = float(word_num)/float(sent_num)  # review length = word number/sentence number
        word_sent_count.append([word_num, sent_num, sent_word])
    return word_sent_count


# Store features
def store_word_sent_num_features(filepath, sheetnum, colnum, data, storepath):
    data = tp.seg_fil_excel(filepath, sheetnum, colnum)

    word_sent_num = word_sent_count(data) # Need initiallized

    f = open(storepath,'w')
    for i in word_sent_num:
        f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
    f.close()
