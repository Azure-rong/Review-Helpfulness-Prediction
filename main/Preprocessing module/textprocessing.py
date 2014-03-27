#! /usr/bin/env python2.7
#coding=utf-8

""" 
Read data from excel file and txt file.
Chinese word segmentation, postagger, sentence cutting and stopwords filtering function.

"""

import xlrd
import jieba
import jieba.posseg
jieba.load_userdict('E:/Python27/Lib/site-packages/jieba-0.31/jieba/userdict.txt') #Load user dictionary to increse segmentation accuracy


"""
input: An excel file with product review
	手机很好，很喜欢。
    三防出色，操作系统垃圾！
    Defy用过3年感受。。。
    刚买很兴奋。当时还流行，机还很贵
    ……
output:
    parameter_1: Every cell is a value of the data list. (unicode)
    parameter_2: Excel row number. (int)
"""
def get_excel_data(filepath, sheetnum, colnum, para):
    table = xlrd.open_workbook(filepath)
    sheet = table.sheets()[sheetnum-1]
    data = sheet.col_values(colnum-1)
    rownum = sheet.nrows
    if para == 'data':
        return data
    elif para == 'rownum':
        return rownum


"""
input:
    parameter_1: A txt file with many lines
    parameter_2: A txt file with only one line of data
output:
    parameter_1: Every line is a value of the txt_data list. (unicode)
    parameter_2: Txt data is a string. (str)
"""

def get_txt_data(filepath, para):
    if para == 'lines':
        txt_file1 = open(filepath, 'r')
        txt_tmp1 = txt_file1.readlines()
        txt_tmp2 = ''.join(txt_tmp1)
        txt_data1 = txt_tmp2.decode('utf8').split('\n')
        txt_file1.close()
        return txt_data1
    elif para == 'line':
        txt_file2 = open(filepath, 'r')
        txt_tmp = txt_file2.readline()
        txt_data2 = txt_tmp.decode('utf8')
        txt_file2.close()
        return txt_data2


"""
input: 这款手机大小合适。
output:
    parameter_1: 这 款 手机 大小 合适 。(unicode)
    parameter_2: [u'\u8fd9', u'\u6b3e', u'\u624b\u673a', u'\u5927\u5c0f', u'\u5408\u9002', u'\uff0c']
"""

def segmentation(sentence, para):
    if para == 'str':
        seg_list = jieba.cut(sentence)
        seg_result = ' '.join(seg_list)
        return seg_result
    elif para == 'list':
        seg_list2 = jieba.cut(sentence)
        seg_result2 = []
        for w in seg_list2:
            seg_result2.append(w)
        return seg_result2


"""
input: '这款手机大小合适。'
output:
    parameter_1: 这 r 款 m 手机 n 大小 b 合适 a 。 x
    parameter_2: [(u'\u8fd9', ['r']), (u'\u6b3e', ['m']),
    (u'\u624b\u673a', ['n']), (u'\u5927\u5c0f', ['b']),
    (u'\u5408\u9002', ['a']), (u'\u3002', ['x'])]
"""

def postagger(sentence, para):
    if para == 'list':
        pos_data1 = jieba.posseg.cut(sentence)
        pos_list = []
        for w in pos_data1:
             pos_list.append((w.word, w.flag)) #make every word and tag as a tuple and add them to a list
        return pos_list
    elif para == 'str':
        pos_data2 = jieba.posseg.cut(sentence)
        pos_list2 = []
        for w2 in pos_data2:
            pos_list2.extend([w2.word.encode('utf8'), w2.flag])
        pos_str = ' '.join(pos_list2)
        return pos_str


"""
input: A review like this
    '这款手机大小合适，配置也还可以，很好用，只是屏幕有点小。。。总之，戴妃+是一款值得购买的智能手机。'
output: A multidimentional list
    [u'\u8fd9\u6b3e\u624b\u673a\u5927\u5c0f\u5408\u9002\uff0c',
    u'\u914d\u7f6e\u4e5f\u8fd8\u53ef\u4ee5\uff0c', u'\u5f88\u597d\u7528\uff0c',
    u'\u53ea\u662f\u5c4f\u5e55\u6709\u70b9\u5c0f\u3002', u'\u603b\u4e4b\uff0c',
    u'\u6234\u5983+\u662f\u4e00\u6b3e\u503c\u5f97\u8d2d\u4e70\u7684\u667a\u80fd\u624b\u673a\u3002']
"""

""" Maybe this algorithm will have bugs in it """
def cut_sentences_1(words):
    #words = (words).decode('utf8')
    start = 0
    i = 0 #i is the position of words
    sents = []
    punt_list = ',.!?:;~，。！？：；～ '.decode('utf8') # Sentence cutting punctuations
    for word in words:
        if word in punt_list and token not in punt_list:
            sents.append(words[start:i+1])
            start = i+1
            i += 1
        else:
            i += 1
            token = list(words[start:i+2]).pop()
    # if there is no punctuations in the end of a sentence, it can still be cutted
    if start < len(words):
        sents.append(words[start:])
    return sents

""" Sentence cutting algorithm without bug, but a little difficult to explain why"""
def cut_sentence_2(words):
    #words = (words).decode('utf8')
    start = 0
    i = 0 #i is the position of words
    token = 'meaningless'
    sents = []
    punt_list = ',.!?;~，。！？；～… '.decode('utf8')
    for word in words:
        if word not in punt_list:
            i += 1
            token = list(words[start:i+2]).pop()
            #print token
        elif word in punt_list and token in punt_list:
            i += 1
            token = list(words[start:i+2]).pop()
        else:
            sents.append(words[start:i+1])
            start = i+1
            i += 1
    if start < len(words):
        sents.append(words[start:])
    return sents


"""
input: An excel file with product reviews
    手机很好，很喜欢。
    三防出色，操作系统垃圾！
    Defy用过3年感受。。。
    刚买很兴奋。当时还流行，机还很贵
output: A multidimentional list of reviews

"""
 
def seg_fil_excel(filepath, sheetnum, colnum):
    # Read product review data from excel file and segment every review
    review_data = []
    for cell in tp.get_excel_data(filepath, sheetnum, colnum, 'data')[0:get_excel_data(filepath, sheetnum, colnum, 'rownum')]:
        review_data.append(segmentation(cell, 'list')) # Seg every reivew
    
    # Read txt file contain stopwords
    stopwords = get_txt_data('D:/code/stopword.txt', 'lines')

    # Filter stopwords from reviews
    seg_fil_result = []
    for review in review_data:
        fil = [word for word in review if word not in stopwords and word != ' ']
        seg_fil_result.append(fil)
        fil = []
 
    # Return filtered segment reviews
    return seg_fil_result


"""
input: An excel file with product reviews
    手机很好，很喜欢。
    三防出色，操作系统垃圾！
    Defy用过3年感受。。。
    刚买很兴奋。当时还流行，机还很贵
output: A multidimentional list of reviews, use different stopword list, so it will remain sentiment tokens.

"""

def seg_fil_senti_excel(filepath, sheetnum, colnum):
    # Read product review data from excel file and segment every review
    review_data = []
    for cell in tp.get_excel_data(filepath, sheetnum, colnum, 'data')[0:get_excel_data(filepath, sheetnum, colnum, 'rownum')]:
        review_data.append(segmentation(cell, 'list')) # Seg every reivew
    
    # Read txt file contain sentiment stopwords
    sentiment_stopwords = get_txt_data('D:/code/seniment_test/sentiment_stopword.txt', 'lines')
 
    # Filter stopwords from reviews
    seg_fil_senti_result = []
    for review in review_data:
        fil = [word for word in review if word not in sentiment_stopwords and word != ' ']
        seg_fil_senti_result.append(fil)
        fil = []
 
    # Return filtered segment reviews
    return seg_fil_senti_result