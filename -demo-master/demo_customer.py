import os
import time
import demo_model
from demo_mysql import get_data
from kafka import KafkaProducer
from kafka import KafkaConsumer
"""
https://blog.csdn.net/weiwangchao_/article/details/81219701
"""
class kafka_consumer:
    def __init__(self, kafkahost, kafkaport, kafkatopic, groupid,date):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.groupid = groupid
        self.consumer = KafkaConsumer(self.kafkatopic, group_id=self.groupid,
                                      bootstrap_servers='{kafka_host}:{kafka_port}'.format(
                                          kafka_host=self.kafkaHost,
                                          kafka_port=self.kafkaPort))
        self.model = demo_model.model()
        self.sql = get_data(date)

    def consume_data(self):
        while True:
            for message in self.consumer:
                data=bytes.decode(message.value)
                label=self.model.predict(data[11])
                data.append(label)
                self.sql.save(data,time.strftime('%Y%m%d',time.localtime(time.time())))
            time.sleep(1)
