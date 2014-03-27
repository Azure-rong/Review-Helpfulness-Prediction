#! /usr/bin/env python2.7
#coding=utf-8

"""
Counting adjective words, adverbs and verbs number in the review.
This module aim to extract adjective words, adverbs and verbs number features.

"""


import textprocessing as tp

# Function of counting review adjectives adverbs and verbs numbers
def count_adj_adv(dataset):
    adj_adv_num = []
    a = 0
    d = 0
    v = 0
    for review in dataset:
        pos = tp.postagger(review, 'list')
        for i in pos:
            if i[1] == 'a':
                a += 1
            elif i[1] == 'd':
                d += 1
            elif i[1] == 'v':
                v += 1
        adj_adv_num.append((a, d, v))
        a = 0
        d = 0
        v = 0
    return adj_adv_num


# Store features
def store_adj_adv_v_num_feature(filepath, sheetnum, colnum, data, storepath):
    data = tp.seg_fil_excel(filepath, sheetnum, colnum)

    adj_adv_num = count_adj_adv(data)

    f = open(storepath,'w')
    for i in adj_adv_num:
        f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
    f.close()