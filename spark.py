
from __future__ import print_function

from pyspark.sql import SparkSession


spark = SparkSession\
 .builder\
 .appName("StructuredKafkaOrnegi")\
.master("local[*]")\
 .getOrCreate()

df = spark\
 .readStream\
 .format("kafka")\
 .option("kafka.bootstrap.servers", "localhost:9092")\
 .option("subscribe", "StatsTest")\
 .load()\
 .selectExpr("CAST(value AS STRING)")

query = df\
 .selectExpr("CAST(value AS STRING)")\
 .writeStream\
 .format('kafka')\
 .option('kafka.bootstrap.servers', 'localhost:9092')\
 .option('topic', 'StatsTestRes')\
 .option('checkpointLocation', '/usr/local/spark/chkpoint/')\
 .start()

query.awaitTermination()