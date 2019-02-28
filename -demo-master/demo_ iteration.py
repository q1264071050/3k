from demo_mysql import GetData
import Classification
import numpy as np

#清洗数据
def clean(text):
    import re

    partern = re.compile("\\[\\d*?\\]")
    partern2 = re.compile("\\<e>\\d*?\\<\/e>")
    partern3 = re.compile("voice.*?content=(.*?)>")
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

    return text
#字符索引转化为向量
def transform(data,word2vec):
    embedding = []
    for i in range(len(data)):
        line = np.zeros((100,))
        for word in data[i].split():
            if word in word2vec:
                # print(word)
                line += word2vec[word]
            embedding.append(line)
    return embedding

#此模型为用sklearn模型的更新方法，若采用聚类、人工打标，则用不上
def upgade_model(table,limit='',test='',word2vec_dir=''):
    '''

    :param table: 客服打标后的准确数量
    :param limit: 无需设计，后续可能开发的字段
    :param test: 测试数据路径
    :param word2vec_dir: 词向量路径
    :return:
    '''

    #读模型
    gbdt = Classification.MYGBDT(model_dir=None, data_dir=None,
                                      word2vec_dir=word2vec_dir)
    gbdt.load()

    #读词向量
    vocab = {}
    with open(word2vec_dir, 'r', encoding='utf-8')as f:
        d = f.readline().strip()
        while d:
            line = d.split()
            vocab[line[0]] = np.array([float(i) for i in line[1:]])
            d = f.readline().strip()
    f.close()

    #读训练数据
    sql = GetData(limit)
    data0=[]
    data1=[]
    for line in sql(table):
        line=list(line)
        if int(line[-1])==1:
            #欲设计的数据结构，第11列为聊天内容
            data1.append(line[11])
        else:
            data0.append(line[11])

    #数据达到一定量和标签相对平衡才更新
    if len(data0)/float(len(data1))<=2 and len(data1)>=5000:

        #加载测试数据
        test_data=[]
        test_label=[]
        with open(test+'data0.txt','r',encoding='utf-8')as f:
            text=f.readline().strip()
            while text:
                words=text.split()
                test_data.append(words)
                test_label.append(0)
                text=f.readline().strip()
        f.close()
        with open(test+'data1.txt','r',encoding='utf-8')as f:
            text=f.readline().strip()
            while text:
                words=text.split()
                test_data.append(words)
                test_label.append(1)
                text=f.readline().strip()
        f.close()
        test_embdding=transform(test_data,vocab)

        #转换训练数据
        import jieba
        jieba.add_word('#MARK#')
        jieba.add_word('手机')
        for i in range(len(data0)):
            text=clean(data0[i])
            words=list(jieba.cut(text))
            data0[i]=words
        for i in range(len(data1)):
            text=clean(data1[i])
            words=list(jieba.cut(text))
            data1[i]=words
        data0=transform(data0,vocab)
        data1=transform(data1,vocab)

        #迭代模型
        gbdt.update(data0+data1,[0]*len(data0)+[1]*len(data1),test_embdding,test_label)

    else:
        return '数据量不足以迭代数据.'
