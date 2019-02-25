import os
import time
import demo_model
from demo_mysql import get_data
from kafka import KafkaProducer
from kafka import KafkaConsumer
"""
https://blog.csdn.net/weiwangchao_/article/details/81219701
"""

# exit(0)
class kafka_producer:
    def __init__(self,kafkahost,kafkaport,kafkatopic,date,limit=""):
        self.host=kafkahost
        self.port=kafkaport
        self.topic=kafkatopic
        self.producer=KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(
            kafka_host=self.host,
            kafka_port=self.port
            ))
        self.sql = get_data(date,limit)

    def send_string(self,):
        producer=self.producer
        for line in self.sql():
            producer.send(self.topic, list(line))
        producer.flush()

