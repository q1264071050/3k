from pyspark import  SparkContext,SparkConf
from pyspark.mllib.clustering import KMeans
import os
import re
import jieba
import numpy as np


def CreateRawClustData(data_dir='/home/second/PycharmProjects/aaa/data/chat/raw_data.csv',
                       word2vec_dir='/home/second/PycharmProjects/aaa/data/chat/chat_word2vec.txt',
                       save='/home/second/PycharmProjects/aaa/data/chat/spark_data.txt'):
    #正则
    partern=re.compile("\\[\\d*?\\]")
    partern2 = re.compile("\\<e>\\d*?\\<\/e>")
    partern3 = re.compile("voice.*?content=(.*?)>")

    #加载词向量
    vocab = {}
    with open(word2vec_dir, 'r', encoding='utf-8')as f:
        d = f.readline().strip()
        while d:
            line = d.split()
            vocab[line[0]] = np.array([float(i) for i in line[1:]])
            # print(vocab[line[0]].shape)
            # exit(0)
            d = f.readline().strip()
    f.close()
    print("word2vec's total number of words is:", len(vocab))

    #读数据，分词，保存
    num = 0
    w = open(save, 'w', encoding='utf-8')
    with open(data_dir, 'r', encoding='utf-8')as f:
        text = f.readline().strip()
        while text:
            try:
                text = text.split(',')[0]
                text2 = partern3.findall(text)

                # 对字符处理
                if len(text2) > 0:
                    if text2[0] != '[语音信息为空]':
                        text = text2[0]
                    else:
                        text = ""
                else:
                    p = partern.findall(text)
                    for pp in p:
                        text = text.replace(pp, "")
                    p = partern2.findall(text)
                    for pp in p:
                        text = text.replace(pp, "#MARK#")
                num += 1
                words = list(jieba.cut(text))
                sum = np.zeros((100,), dtype=np.float)
                miss = 0
                line=""
                for each in words:
                    try:
                        sum += vocab[each]
                        line=line+" "+each
                    except:
                        miss += 1
                if np.sum(sum) != 0:
                    sum = [str(i) for i in (sum)]
                    w.write(" ".join(sum) + ',' +line+'\n')
                    print(line)
                # print(data)
                # exit(0)
                text = f.readline().strip()
            except:
                text = f.readline().strip()
    f.close()
    w.close()
    print("total number of data is:", num)


def parse_interaction(line):
    """
    Parses a network data interaction.
    """
    line_split = line.split(",")
    clean_line_split = line_split[0].split()
    return ("".join(line_split[1]), np.array([float(x) for x in clean_line_split]))
def SparkKmeans(data_file='/home/second/PycharmProjects/aaa/data/chat/spark_data.txt'):
    # data_file='test.txt'
    conf = SparkConf().setAppName("cluster")
    sc = SparkContext(conf=conf)
    raw_data = sc.textFile(data_file)
    labels = raw_data.map(lambda line: line.strip().split(",")[1])
    print ("Parsing dataset...")
    parsed_data = raw_data.map(parse_interaction)
    standardized_data_values = parsed_data.values().cache()
    # Standardize data
    # print ("Standardizing data...")
    # standardizer = StandardScaler(True, True)
    # standardizer_model = standardizer.fit(parsed_data_values)
    # standardized_data_values = standardizer_model.transform(parsed_data_values)
    print ("training dataset...")
    clusters = KMeans.train(standardized_data_values, 500, maxIterations=10, runs=5, initializationMode="random")
    cluster_assignments_sample = standardized_data_values.map(lambda datum: str(clusters.predict(datum))+","+",".join(map(str,datum)))
    print ("saving result...")
    cluster_assignments_sample.saveAsTextFile("standardized")
    labels.saveAsTextFile("labels")

def merge_data(base='/home/second/PycharmProjects/aaa/chat_demo/',save='/home/second/PycharmProjects/aaa/chat_demo/cluster/'):
    labels_base=base+"labels/"
    stand_base=base+"standerized/"
    data={}
    for file in os.listdir(labels_base):
        if "part-" in file:
            f1=open(stand_base+file,'r',encoding='utf-8')
            f2 = open(labels_base+file, 'r', encoding='utf-8')
            d1=f1.readline().strip()
            d2=f2.readline().strip()
            while (d1 and d2):
                d1=d1.split(',')
                if d1[0] not in data:
                    data[d1[0]]=[]
                    data[d1[0]].append(d2)
                else:
                    data[d1[0]].append(d2)
                d1=f1.readline().strip()
                d2 = f2.readline().strip()
            f1.close()
            f2.close()
    print(len(data))
    for c in data:
        with open(save+"cluster_"+str(c)+".txt",'w',encoding='utf-8') as f:
            for d in data[c]:
                f.write(d+'\n')
        f.close()



