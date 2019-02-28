import os
import time
import demo_model
from demo_mysql import GetData
from kafka import KafkaProducer
from kafka import KafkaConsumer
"""
https://blog.csdn.net/weiwangchao_/article/details/81219701
"""


class kafka_producer:
    def __init__(self,kafkahost,kafkaport,kafkatopic,limit=""):
        '''

        :param kafkahost: 生产者所在ip
        :param kafkaport: 生产者端口
        :param kafkatopic: 生产者主题
        :param limit: 限制条件，选择表后的筛选条件
        '''
        self.host=kafkahost
        self.port=kafkaport
        self.topic=kafkatopic
        self.producer=KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(
            kafka_host=self.host,
            kafka_port=self.port
            ))
        self.sql = GetData(limit)

    #发送数据
    def send_string(self,table):
        '''
        :param table:选择要读取数据的表
        :return:
        '''
        producer=self.producer
        #遍历数据，发送
        for line in self.sql(table):
            producer.send(self.topic, ",".join(list(line)))
        producer.flush()

