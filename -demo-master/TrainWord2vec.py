import jieba
import gensim.models
import csv
import gc
import re
import multiprocessing
import numpy as np
import sys
jieba.add_word("#MARK#")
jieba.add_word("手机")
csv.field_size_limit(sys.maxsize)


def GroupDataToWord2vec(data_dir='/home/second/PycharmProjects/aaa/data/chat/raw_data.csv',
                        save='/home/second/PycharmProjects/aaa/data/chat/word2vec_train_data.txt'):

    #正则
    partern=re.compile("\\[\\d*?\\]")
    partern2 = re.compile("\\<e>\\d*?\\<\/e>")
    partern3 = re.compile("voice.*?content=(.*?)>")

    #读数据
    w = open(save, 'w', encoding='utf-8')
    file=open(data_dir,'r',encoding='utf-8')
    reader=csv.reader(file)

    #分组用
    mark=0
    s=""

    #遍历数据
    for num,text in enumerate(reader):
        if len(text)==0:
            continue
        label=text[1]
        text=text[0]
        text2 = partern3.findall(text)

        #对字符处理
        if len(text2) > 0:
            if text2[0] != '[语音信息为空]':
                text = text2[0]
            else:
                text=""
        else:

            p = partern.findall(text)
            for pp in p:
                text = text.replace(pp, "")
            p = partern2.findall(text)
            for pp in p:
                text = text.replace(pp, "#MARK#")

        #分组,分词
        if num==0:
            mark=label
            s=text.strip()
        else:
            try:
                if label==mark:
                    if len(text)==0:
                        continue
                    s=s+" "+text
                else:
                    line = ""
                    words = list(jieba.cut(s))
                    for index, word in enumerate(words):
                        if index == 0:
                            line = word
                        elif word != " ":
                            line = line + " " + word.strip()
                    if len(line) >= 4:
                        print(line)
                        w.write(line + '\n')
                    s=text
            except:
                continue
    file.close()

    #处理最后一行
    line = ""
    words = list(jieba.cut(s))
    for index, word in enumerate(words):
        if index == 0:
            line = word
        elif word != " ":
            line = line + " " + word.strip()
    if len(line) >= 4:
        print(line)
        w.write(line + '\n')
    w.close()
def word2vec(data_dir='/home/second/PycharmProjects/aaa/data/chat/word2vec_train_data.txt',
             save='/home/second/PycharmProjects/aaa/data/chat/chat_word2vec.txt'):
    #读模型
    # model=gensim.models.word2vec.Word2Vec.load("./model/chat_w2v_2.model")

    #读数据
    sentences = gensim.models.word2vec.Text8Corpus(data_dir)
    #训练模型
    model=gensim.models.word2vec.Word2Vec(sentences,size=100,min_count=10,workers=multiprocessing.cpu_count())
    #保存模型
    model.save("/home/second/PycharmProjects/aaa/data/chat/model/chat_w2v.model")

    #加载数据集所有词
    wordemb = set()
    with open(data_dir,'r',encoding='utf-8')as f:
        d=f.readline().strip()
        while d:
            words=d.split()
            for each in words:
                wordemb.add(each)
            d=f.readline().strip()
    f.close()
    print("the number of unique words in dataset is:",len(wordemb))

    #保存为txt文件
    w = open(save, 'w', encoding='utf-8')
    for each in wordemb:
        if each in model:
            line = list(model[each])
            lines = [str(i) for i in line]
            linestr = ' '.join(lines)
            L = each + ' ' + linestr
            w.write(L+"\n")
    w.close()







#手写的聚类，用spark代替
def cos_sim(a,b):
    return 0.5+0.5*(float(np.matmul(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))
def clusters(data_dir='/home/second/PycharmProjects/aaa/data/chat/data_cluster2.txt'):

    vocab={}
    with open('/home/second/PycharmProjects/aaa/data/chat_w2v_190220.txt','r',encoding='utf-8')as f:
        d=f.readline().strip()
        while d:
            line=d.split()
            vocab[line[0]]=np.array([float(i) for i in line[1:]])
            # print(vocab[line[0]].shape)
            # exit(0)
            d=f.readline().strip()
    f.close()
    print(len(vocab))
    data = {}
    num = 0
    with open(data_dir, 'r', encoding='utf-8')as f:
        d = f.readline().strip()
        while d:
            try:
                #d = d.split('-*-')
                num += 1
                words = d.split()
                sum = np.zeros((100,), dtype=np.float)
                miss = 0
                for each in words:
                    # print(vocab[each])
                    try:
                        sum += vocab[each]
                    except:
                        miss += 1
                if np.sum(sum) < 0:
                    data[d] = sum
                # print(data)
                # exit(0)
                d = f.readline().strip()
            except:
                d = f.readline().strip()
    f.close()
    print(num)
    print(len(data))
    del vocab
    gc.collect()
    threshold = 0.6
    number = 0
    classes = {}
    for num, d in enumerate(data):
        if num == 0:
            classes[number] = [dict(), data[d]]
            classes[number][0][d] = data[d]
            number += 1
        else:
            max = 0.
            c = number
            for dd in classes:
                result = cos_sim(data[d], classes[dd][1])
                if result >= max:
                    max = result
                    c = dd
            if max < threshold:
                c = number
            if c not in classes:
                classes[number] = [dict(), data[d]]
                classes[number][0][d] = data[d]
                print(str(num+1)+'/'+str(len(data)),d, c)
                number += 1
            else:
                classes[c][0][d] = data[d]
                print(str(num+1)+'/'+str(len(data)),d, c)
                arg = np.zeros((100,), dtype=np.float)
                for sent in classes[c][0]:
                    # print(sent[1][0])
                    arg += classes[c][0][sent]
                arg = arg / float(len(classes[c][0]))
                classes[c][1] = arg
    del data
    gc.collect()
    for c in classes:
        with open("./data/c3/class"+str(c)+".txt",'w',encoding='utf-8')as w:
            for d in classes[c][0]:
                w.write(d+"\n")
        w.close()
    print(len(classes))
clusters()
