import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from confluent_kafka import Producer
from confluent_kafka import Consumer, KafkaException, KafkaError
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import IntegerType, DoubleType
import joblib
from pyspark.ml.feature import VectorAssembler


import json
data = pd.read_csv('C:\\Users\\Pc\\OneDrive\\Desktop\\StudentGradesAndPrograms.csv')



# Eksik değer sayısını kontrol etme
#print(data.isnull().sum())
#print(data)

## Eksik değer bulunmamaktadır.
## Anormallik tespitinde kullanılmayacak sütunlar dataset işleme kısmından çıkarılmaktadır.
df= data.drop(['schoolyear', 'gradeLevel','classPeriod','classType','schoolName','sped'], axis='columns')

# Anormallik tespitinde kullanılacak gradePercentage değeri 0-2k aralığından 0-2 aralığına çekerek veri normalize edilmiştir.

scaler = MinMaxScaler(feature_range=(0, 2))  # Veriyi 0 ile 2 arasında normalize et
df['gradePercentage'] = scaler.fit_transform(df[['gradePercentage']])

#Student_id sütunu harf ve rakamlardan oluşmaktadır. Encoding gereği vbu sütun ASCII dönüşümü yapılarak tamamen sayısal hale getirilmiştir.
def encode_to_single_number(student_id):
    encoded = ''.join([str(ord(char)) for char in student_id])  
    return int(encoded) 
df['student_ID'] = df['student_ID'].apply(encode_to_single_number)


# Anormallik tespitinde kullanacağımız ell,migrant,avid sütunundaki Y(yes) ve N(no) değerleri encosing ile sayısal değere dönüştürülmüştür.
#Y->1
#N->0
le = LabelEncoder()
df['ell'] = le.fit_transform(df['ell'])
df['migrant'] = le.fit_transform(df['migrant'])
df['avid'] = le.fit_transform(df['avid'])
#print(df)



#-AVID(Advancement Via Individual Determination): Fırsat eşitsizliğini kapatmaya odaklanmış bir eğitim programı 
#ELL(English Language Learner) : İngilizceyi ek bir dil olarak öğrenen öğrenciler için bir program.
#Migrant:Göçmen olarak tanımlanan öğrenciler için destek programları. Bir çocuk, ebeveyn veya vasisinin tarım, 
# süt ürünleri, kereste veya balıkçılık sektörlerinde geçici işçi olması ve ailesinin son üç yıl içinde yer 
# değiştirmiş olması durumunda "göçmen" olarak kabul edilir.
# AVİD , ELL programlarına katılan ve göçmen olup destek porgramıjndan yararlanmasıan rağmen 
#GradePercentAge değeri(çıkarım anındaki sınıf puanı ) 2000'lik sistemde 1000'i geçememiş öğrencilerin bu programlardan
#men edilme amacıyla tespit edilmesi gerekmektedir. Bu projede bu amaç anormallik varsayılarak işlem yapılacaktır.

#Outlier tespiti ve işlenmesi


# Anormal durumları tespit etme: 'avid','ell' ve 'migrant' değeri 1 olan ve 'gradePercentage' değeri 1'den küçük olan satırlar
df['is_anomaly'] = ((df['avid'] == 1) & (df['ell'] == 1) &(df['migrant'] == 1) & (df['gradePercentage'] < 1.00)).astype(int)

# Anormal durumları yazdırma
#print("\nAnormal durumlar:")
#print(df[df['is_anomaly'] == 1])
anomalies = df[df['is_anomaly'] == 1] 
normalies= df[df['is_anomaly'] !=1] 
# Anormal durumları işleme: Örneğin, 'gradePercentage' değerini 1'e eşitleme
#df.loc[df['is_anomaly'] == 1, 'gradePercentage'] = 1.1


# Güncellenmiş veri çerçevesini yazdırma
#print("\nGüncellenmiş veri:")
print(df[df['is_anomaly'] == 1])

# VERİ GÖRSELLEŞTİRMELERİ

# Histogram: Veri dağılımını görmek için
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Veri Dağılımı - Histogram", fontsize=16)
plt.show()

# Boxplot: Outlier analizi için
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient="h")
plt.title("Boxplot ile Outlier Tespiti", fontsize=16)
plt.show()

# Korelasyon Matrisi: Özellikler arasındaki ilişkiyi incelemek için
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi", fontsize=16)
plt.show()

# Scatter plot: GradePercentage'a göre anomalilerin gösterimi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="gradePercentage", y="is_anomaly", hue="is_anomaly", palette="coolwarm", s=100)
plt.title("Anomalilerin Görselleştirilmesi", fontsize=16)
plt.xlabel("GradePercentage")
plt.ylabel("Is Anomaly")
plt.legend(title="Anomaly", loc="upper right")
plt.show()

# Yapay zeka modeli eğitilmesi


X = df[['ell', 'migrant', 'avid', 'gradePercentage']]  
y = df['is_anomaly'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Random Forest Modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 2. Decision Tree Modeli
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# 3. Logistic Regression Modeli
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)


models = {
    'Random Forest': rf_model,
    'Decision Tree': dt_model,
    'Logistic Regression': lr_model
}

predictions = {
    'Random Forest': y_pred_rf,
    'Decision Tree': y_pred_dt,
    'Logistic Regression': y_pred_lr
}

#Modellerin Raporlanması

# Random Forest Modelinin Performansı
print("\nRandom Forest Modeli Performansı:")
print("Accuracy:", accuracy_score(y_test, predictions['Random Forest']))
print("Precision:", precision_score(y_test, predictions['Random Forest']))
print("Recall:", recall_score(y_test, predictions['Random Forest']))
print("F1 Score:", f1_score(y_test, predictions['Random Forest']))

# Classification Report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, predictions['Random Forest']))


# Decision Tree Modelinin Performansı
print("\nDecision Tree Modeli Performansı:")
print("Accuracy:", accuracy_score(y_test, predictions['Decision Tree']))
print("Precision:", precision_score(y_test, predictions['Decision Tree']))
print("Recall:", recall_score(y_test, predictions['Decision Tree']))
print("F1 Score:", f1_score(y_test, predictions['Decision Tree']))

# Classification Report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, predictions['Decision Tree']))


# Logistic Regression Modelinin Performansı
print("\nLogistic Regression Modeli Performansı:")
print("Accuracy:", accuracy_score(y_test, predictions['Logistic Regression']))
print("Precision:", precision_score(y_test, predictions['Logistic Regression']))
print("Recall:", recall_score(y_test, predictions['Logistic Regression']))
print("F1 Score:", f1_score(y_test, predictions['Logistic Regression']))

# Classification Report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, predictions['Logistic Regression']))

"""
#Kafka Prodeucer
#Tespit edilen anormallikleri Kafka'ya göndermek için kullanıyoruz.


# Kafka Producer konfigürasyonu
conf = {
    'bootstrap.servers': 'localhost:9092',  # Kafka sunucusu adresi
    'queue.buffering.max.messages': 1000000,  # Kuyruğun kapasitesini arttırma
    'queue.buffering.max.kbytes': 1048576,    # Kuyruğun byte cinsinden kapasitesini arttırma
    'batch.num.messages': 1000,               # Kafka'ya bir seferde gönderilecek mesaj sayısı
    'linger.ms': 1000,                        # Mesajlar birikene kadar bekleme süresi (ms)
}
producer = Producer(conf)

def send_to_kafka(topic, anomalies_df, num_records=1000):
    try:
        # Verinin sadece ilk 'num_records' kadarını alıyoruz
        anomalies_subset = anomalies_df.head(num_records)

       
        for _, row in anomalies_subset.iterrows():
            message = row.to_dict()  
            producer.produce(topic, value=json.dumps(message))  
        producer.flush() 
        print(f"{len(anomalies_subset)} adet veri başarıyla '{topic}' konusuna gönderildi.")
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
def send_to_kafka(topic, normalies_df, num_records=1000):
    try:
       
        normalies_subset = normalies_df.head(num_records)

   
        for _, row in normalies_subset.iterrows():
            message = row.to_dict()  
            producer.produce(topic, value=json.dumps(message))  
        producer.flush() 
        print(f"{len(normalies_subset)} adet veri başarıyla '{topic}' konusuna gönderildi.")
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")


send_to_kafka('anomaly_topic', anomalies,num_records=1000)
send_to_kafka('normal_topic',normalies,num_records=1000)

def delivery_report(err, msg):

    #Callback fonksiyonu: Mesajın başarıyla gönderilmesi durumunda çalışır
    
    if err is not None:
        print(f"Mesaj gönderilemedi: {err}")
    else:
        print(f"Mesaj başarıyla gönderildi: {msg.topic} [{msg.partition}] @ {msg.offset}")

# Kafka topic adı (Verinin gönderileceği topic)
topic_name = 'input_topic'

# Veriyi Kafka'ya göndermek
count = 0
for _, row in df.iterrows():
    if count >= 10:
        break  
    record = row.to_dict()  # Her bir satırı dict formatına çeviriyoruz
    producer.produce(topic_name, value=json.dumps(record), callback=delivery_report)
    count += 1
# Mesajların gönderilmesini bekleme
producer.flush()


# Spark session başlatma
spark = SparkSession.builder \
    .appName("AnomalyDetection") \
.master("local[*]")\
    .getOrCreate()

joblib.dump(rf_model, "random_forest_model.pkl")
# Spark ve Kafka entegrasyonu ile model kullanımı
# Kafka'dan veri okuma
kafka_stream = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "input_topic") \
    .load()\
    .selectExpr("CAST(value AS STRING)")

# Kafka'dan gelen mesajları işleme
df_stream = kafka_stream.selectExpr("CAST(value AS STRING)")
json_df = df_stream.selectExpr(
    "from_json(value, 'ell DOUBLE, migrant DOUBLE, avid DOUBLE, gradePercentage DOUBLE') as data"
).select("data.*")

# Özellikleri vektöre dönüştürme
features = ['ell', 'migrant', 'avid', 'gradePercentage']
assembler = VectorAssembler(inputCols=features, outputCol='features')
vectorized_df = assembler.transform(json_df)

# Pandas DataFrame'e dönüştürme ve model yükleme
def predict_with_rf(data):
    pandas_df = data.toPandas()
    rf_model = joblib.load("random_forest_model.pkl")  # Modeli yükleme
    predictions = rf_model.predict(pandas_df[features])  # Tahmin
    pandas_df['prediction'] = predictions
    return pandas_df

# Akış işlemi
def process_and_send_to_kafka(batch_df, batch_id):
    pandas_predictions = predict_with_rf(batch_df)
    for _, row in pandas_predictions.iterrows():
        topic = 'anomaly_topic' if row['prediction'] == 1 else 'normal_topic'
        producer.send(topic, {'prediction': row['prediction']})

# Streaming işlemi başlatma
query = vectorized_df.writeStream \
    .foreachBatch(process_and_send_to_kafka) \
    .outputMode("update") \
    .start()

# Akışı bekletme
query.awaitTermination()"""