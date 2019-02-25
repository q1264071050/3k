from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import numpy as np
import random
import pickle
import jieba
import re
jieba.add_word("#MARK#")
jieba.add_word("手机")


def divide_data(data0,label0,data1,label1,word2vec):
    cut0=int(len(data0)*2/3)
    cut1=int(len(data1)*2/3)
    train_data=data0[:cut0]+data1[:cut1]
    train_label=label0[:cut0]+label1[:cut1]
    test_data=data0[cut0+1:]+data1[cut1+1:]
    test_label=label0[cut0+1:]+label1[cut1+1:]

    sd0=list(zip(train_data,train_label))
    random.shuffle(sd0)
    train_data,train_label=zip(*sd0)
    sd1=list(zip(test_data,test_label))
    random.shuffle(sd1)
    test_data,test_label=zip(*sd1)
    train_data=np.array(train_data)
    train_label=np.array(train_label)
    test_data=np.array(test_data)
    test_label=np.array(test_label)

    #索引词向量，作全加处理
    train=[]
    for i in range(len(train_data)):
        line = np.zeros((100,))
        for word in train_data[i].split():
            if word in word2vec:
                line += word2vec[word]
        train.append(line)
    test=[]
    for i in range(len(test_data)):
        line = np.zeros((100,))
        for word in train_data[i].split():
            if word in word2vec:
                line += word2vec[word]
        test.append(line)
    print(len(train_data),len(train_label))
    print(len(test_data),len(test_label))
    return train,train_data,train_label,test,test_data,test_label
def CreateData(base="/home/second/PycharmProjects/aaa/chat_demo/cluster/",
               word2vec_dir='/home/second/PycharmProjects/aaa/data/chat/chat_word2vec.txt',
               save='/home/second/PycharmProjects/aaa/data/chat/classification/data.pkl'):

    data_file=[]
    f=open(base+'label.csv','r',encoding='utf-8')
    d=f.readline().strip()
    while d:
        data_file.append(d.split(','))
        d=f.readline().strip()
    f.close()


    # 加载词向量
    vocab = {}
    with open(word2vec_dir, 'r', encoding='utf-8')as f:
        d = f.readline().strip()
        while d:
            line = d.split()
            vocab[line[0]] = np.array([float(i) for i in line[1:]])
            d = f.readline().strip()
    f.close()
    print("word2vec's total number of words is:", len(vocab))

    #获取标签数据
    data0=[]
    label0=[]
    data1=[]
    label1=[]
    # 读数据，分词
    for line in data_file:
        with open(base+line[0], 'r', encoding='utf-8')as f:
            if int(line[1])==0:
                d = f.readline().strip()
                while d:
                    words = d.split()
                    data0.append(words)
                    label0.append(0)
                    d = f.readline().strip()
            else:
                d = f.readline().strip()
                while d:
                    words = list(jieba.cut(d))
                    data1.append(words)
                    label1.append(1)
                    d = f.readline().strip()
        f.close()

    #根据词向量，构造数据集，打乱处理
    train_data,train_string,train_label,test_data,test_string,test_label=divide_data(data0,label0,data1,label1,vocab)
    file=open(save,'wb')
    pickle.dump([train_data,train_string,train_label,test_data,test_string,test_label],file)
    file.close()

class MYGBDT:
    def __init__(self,model_dir=None,data_dir=None,word2vec_dir='/home/second/PycharmProjects/aaa/data/chat/chat_word2vec.txt'):
        self.moder_dir=model_dir
        self.data_dir=data_dir
        self.model=None
        self.acc=0.
        self.word2vec_dir = word2vec_dir

    def load(self):
        assert (self.moder_dir is not None and self.word2vec_dir is not None)
        try:
            f=open('/home/second/PycharmProjects/aaa/data/chat/classification/model/acc.txt','r',encoding='utf-8')
            self.acc=float(f.readline().strip())
            f.close()
        except:
            print('can\'t load the model\'s acc.')
        self.vocab={}
        with open(self.word2vec_dir, 'r', encoding='utf-8')as f:
            d = f.readline().strip()
            while d:
                line = d.split()
                self.vocab[line[0]] = np.array([float(i) for i in line[1:]])
                d = f.readline().strip()
        f.close()
        self.model=joblib.load(self.moder_dir)

    def train(self):
        assert self.data_dir is not None
        file=open(self.data_dir,'rb')
        [train_data, train_string, train_label, test_data, test_string, test_label]=pickle.load(file)
        file.close()
        model = GradientBoostingClassifier()
        model.fit(train_data, train_label)
        self.model=model
        joblib.dump(model, '/home/second/PycharmProjects/aaa/data/chat/classification/model/gbdt.model')
        f=open('/home/second/PycharmProjects/aaa/data/chat/classification/model/acc.txt','w',encoding='utf-8')
        f.write(str(self.acc)+'\n')
        f.close()
        pre_y = model.predict(test_data)
        acc = 0.
        for i in range(len(pre_y)):
            if pre_y[i] == test_label[i]:
                acc += 1
        self.acc=acc
        print('the model\'s acc is:',acc / float(len(test_label)))

    def predict(self,text):
        assert self.model is not None
        partern2 = re.compile("\\<e>\\d*?\\<\/e>")
        p = partern2.findall(text)
        for pp in p:
            text = text.replace(pp, "#MARK#")
        words=list(jieba.cut(text))
        embedding = np.zeros((100,))
        for word in words:
            if word in self.vocab:
                embedding += self.vocab[word]
        label=self.model.predict(embedding)
        return label[0]

    def update(self):
        assert self.model is not None and self.data_dir is not None
        file = open(self.data_dir, 'rb')
        [train_data, train_string, train_label, test_data, test_string, test_label] = pickle.load(file)
        self.model.fit(train_data, train_label)
        pre_y = self.model.predict(test_data)
        acc = 0.
        for i in range(len(pre_y)):
            if pre_y[i] == test_label[i]:
                acc += 1
        if acc>self.acc:
            joblib.dump(self.model, '/home/second/PycharmProjects/aaa/data/chat/classification/model/gbdt.model')
            self.acc=acc
            f = open('/home/second/PycharmProjects/aaa/data/chat/classification/model/acc.txt', 'w', encoding='utf-8')
            f.write(str(self.acc) + '\n')
            f.close()