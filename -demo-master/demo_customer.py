import os
import time
import Classification
from demo_mysql import GetData
from kafka import KafkaProducer
from kafka import KafkaConsumer
"""
https://blog.csdn.net/weiwangchao_/article/details/81219701
"""
class kafka_consumer:
    def __init__(self, model_dir,word2vec_dir,kafkahost, kafkaport, kafkatopic, groupid,limit='',data_dir=None):
        '''

        :param model_dir: 模型路径
        :param data_dir: 训练数据；实时模型不需要
        :param word2vec_dir: 词向量文件路径
        :param kafkahost: 要获取数据的生产者所在ip
        :param kafkaport: 要获取数据的生产者所在ip访问端口
        :param kafkatopic: 要获取数据的主题
        :param groupid: 消费者组名字；可以随便起
        :param limit: 限制条件；实时模型不需要
        '''

        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.groupid = groupid

        #创建消费者
        self.consumer = KafkaConsumer(self.kafkatopic, group_id=self.groupid,
                                      bootstrap_servers='{kafka_host}:{kafka_port}'.format(
                                          kafka_host=self.kafkaHost,
                                          kafka_port=self.kafkaPort))

        #加载模型
        self.gbdt = Classification.MYGBDT(model_dir=model_dir,data_dir=data_dir,word2vec_dir=word2vec_dir)
        self.gbdt.load()
        #初始化连接数据库
        self.sql = GetData(limit)

    def consume_data(self,table):
        '''

        :param table:保存预测数据的表
        :return:
        '''
        while True:
            #一条条数据处理
            for message in self.consumer:
                #从生产者获取数据
                data=bytes.decode(message.value).split(',')
                #预测标签
                label=self.gbdt.predict(data[11])
                #在数据后拼接
                data.append(label)
                #保存到数据库表中
                self.sql.save(data,table)
            #每个一秒检查一次生成者
            time.sleep(1)
