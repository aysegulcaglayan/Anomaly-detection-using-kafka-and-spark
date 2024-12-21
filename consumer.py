from confluent_kafka import Consumer, KafkaException, KafkaError
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
import json

# Kafka Consumer konfigürasyonu
conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my_consumer_group',
    'auto.offset.reset': 'earliest'
    
}

consumer = Consumer(conf)
consumer.subscribe(['anomaly_topic','normal_topic'])

def consume_from_kafka():
    try:
        while True:
            msg = consumer.poll(timeout=1.0)  
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    raise KafkaException(msg.error())
            else:
               
              # Mesaj tipini kontrol etme
                topic = msg.topic()
                message = json.loads(msg.value().decode('utf-8'))

                if topic == 'anomaly_topic':
                    print('Anomali verisi alındı:', message)
                elif topic == 'normal_topic':
                    print('Normal verisi alındı:', message)
                
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


consume_from_kafka()
