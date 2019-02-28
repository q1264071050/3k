from sklearn.ensemble import GradientBoostingRegressor,RandomForestClassifier
from sklearn.externals import joblib
# from sklearn.svm import SVR
import numpy as np
import random
import pickle
import jieba
import re
jieba.add_word("#MARK#")
jieba.add_word("手机")

#切分数据
def divide_data(data0,label0,data1,label1,word2vec):
    random.shuffle(data0)
    random.shuffle(data1)
    cut0=int(len(data0)*2/3)
    cut1=int(len(data1)*2/3)
    print(cut0,len(data0))
    print(cut1,len(data1))
    train_data=data0[:cut0]+data1[:cut1]
    train_label=label0[:cut0]+label1[:cut1]
    test_data=data0[cut0:]+data1[cut1:]
    test_label=label0[cut0:]+label1[cut1:]
    print(len(train_data))
    print(len(test_data))

    sd0=list(zip(train_data,train_label))
    random.shuffle(sd0)
    train_data,train_label=zip(*sd0)
    sd1=list(zip(test_data,test_label))
    random.shuffle(sd1)
    test_data,test_label=zip(*sd1)
    # train_data=np.array(train_data)
    train_label=np.array(train_label)
    # test_data=np.array(test_data)
    test_label=np.array(test_label)
    print(len(train_data))
    print(len(test_data))

    #索引词向量，作全加处理
    train=[]
    for i in range(len(train_data)):
        line = np.zeros((100,))
        for word in train_data[i].split():
            if word in word2vec:
                # print(word)
                line += word2vec[word]

        train.append(line)
    test=[]
    for i in range(len(test_data)):
        line = np.zeros((100,))
        for word in train_data[i].split():
            if word in word2vec:
                # print(word)
                line += word2vec[word]
        test.append(line)
    print(len(train_data),len(train_label))
    print(len(test_data),len(test_label))
    return train,train_data,train_label,test,test_data,test_label

#生成向量形式的数据
def CreateData(data_dir="/home/second/PycharmProjects/aaa/chat_demo/cluster/",
               word2vec_dir='/home/second/PycharmProjects/aaa/data/chat_w2v_190220.txt',
               save='/home/second/PycharmProjects/aaa/chat_demo/data/data.pkl'):
    '''
    :param data_dir: 两个数据的路径；（名字默认为data0.txt,data1.txt分别是拉人数据）格式为每行以空格隔开的词；
    :param word2vec_dir:词向量路径
    :param save:模型保存路径
    :return:
    '''

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
    data_file=open(data_dir+'data0.txt','r',encoding='utf-8')
    line=data_file.readline().strip()
    while line:
        data0.append(line)
        label0.append(0)
        line=data_file.readline().strip()
    data_file.close()
    data_file = open(data_dir+'data1.txt', 'r', encoding='utf-8')
    line = data_file.readline().strip()
    while line:
        data1.append(line)
        label1.append(1)
        line = data_file.readline().strip()
    data_file.close()
    print('finish')

    #根据词向量，构造数据集，打乱处理
    train_data,train_string,train_label,test_data,test_string,test_label=divide_data(data0,label0,data1,label1,vocab)
    file=open(save,'wb')
    pickle.dump([train_data,train_string,train_label,test_data,test_string,test_label],file)
    file.close()


class MYGBDT:
    def __init__(self,model_dir=None,data_dir=None,word2vec_dir='/home/second/PycharmProjects/aaa/data/chat/chat_word2vec.txt'):
        '''

        :param model_dir: 模型路径
        :param data_dir: 训练数据路径
        :param word2vec_dir: 词向量路径
        '''
        self.moder_dir=model_dir
        self.data_dir=data_dir
        self.model=None
        self.acc=0.
        self.word2vec_dir = word2vec_dir

    #加载现有模型
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

    #训练模型
    def train(self):
        assert self.data_dir is not None
        file=open(self.data_dir,'rb')
        [train_data, train_string, train_label, test_data, test_string, test_label]=pickle.load(file)
        file.close()
        model = RandomForestClassifier()
        model.fit(train_data, train_label)
        self.model=model
        joblib.dump(model, '/home/second/PycharmProjects/aaa/chat_demo/data/model/gbdt.model')
        f=open('/home/second/PycharmProjects/aaa/chat_demo/data/model/acc.txt','w',encoding='utf-8')
        f.write(str(self.acc)+'\n')
        f.close()
        pre_y = model.predict(test_data)
        acc = 0.
        file0=open('/home/second/PycharmProjects/aaa/chat_demo/data/model/data0.txt','w',encoding='utf-8')
        # file1=open('/home/second/PycharmProjects/aaa/chat_demo/data/model/data1.txt','w',encoding='utf-8')

        for i in range(len(pre_y)):
            x=pre_y[i]
            if x>=0.5:
                x=1
            else:
                x=0
            if x == test_label[i]:
                if x==0:
                    file0.write(test_string[i]+'\n')
                # else:
                #     file1.write(test_string[i] + '\n')
                acc += 1
        file0.close()
        # file1.close()
        self.acc=acc
        print('the model\'s acc is:',acc / float(len(test_label)))

    #预测数据
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

    #迭代更新模型
    def update(self,train_data,train_label,test_data,test_label):
        assert self.model is not None and self.data_dir is not None
        file = open(self.data_dir, 'rb')
        # [train_data, train_string, train_label, test_data, test_string, test_label] = pickle.load(file)
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

# model=MYGBDT(data_dir='/home/second/PycharmProjects/aaa/chat_demo/data/data.pkl')
# model.train()