import findspark
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from joblib import dump, load
import pandas as pd
findspark.init()
from pyspark.ml.clustering import KMeansModel


class K_mean_cluster:
    def __init__(self, file_name_1=None,file_name_2= None):
        self.__file_name_1 = file_name_1
        self.__file_name_2 = file_name_2
        self.__spark = SparkSession.builder.master("local[*]").appName("K_mean").config(
            "spark.sql.debug.maxToStringFields", 1000000).config("spark.executor.memory", "4g").config("spark.driver.memory", "2g").getOrCreate()

    def data_clear(self):
        try:
            # đọc dữ liệu vào từ tệp vào df
            df_1 = self.__spark.read.option("delimter", "\t").csv(self.__file_name_1, header=True,
                                                                       inferSchema=True)
            df_2 = self.__spark.read.option("delimter","\t").csv(self.__file_name_2,header=True,inferSchema=True)

            df = df_1.join(df_2,on =['Customer ID', 'Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Location',
                 'Size', 'Color', 'Season', 'Review Rating', 'Subscription Status', 'Shipping Type', 'Discount Applied',
                 'Promo Code Used', 'Previous Purchases', 'Payment Method', 'Frequency of Purchases'],how='inner')
            # df_count =df.count()
            # df.printSchema()
            # # print(df_count) 15210000
            distinct_columns = df.dropDuplicates(['Customer ID', 'Age', 'Gender', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Location',
                 'Size', 'Color', 'Season', 'Review Rating', 'Subscription Status', 'Shipping Type', 'Discount Applied',
                 'Promo Code Used', 'Previous Purchases', 'Payment Method', 'Frequency of Purchases'])
            # distinct_columns.printSchema()
            # distinct_columns.show(5)
            feature_col =['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
            # Tiền xử lý dữ liệu: chuyển đổi các thuộc tính phân loại thành dạng số
            indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(distinct_columns) for col in ['Gender']]
            df_indexed = distinct_columns
            for indexer in indexers:
                df_indexed = indexer.transform(df_indexed)
            assembler = VectorAssembler(inputCols=feature_col,outputCol="features")
            df_assembler= assembler.transform(df_indexed)
            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            train_data, test_data = df_assembler.randomSplit([0.8, 0.2], seed=42)
            # Khởi tạo mảng để lưu inertia
            inertias = []

            # Thử nghiệm số lượng cụm từ 2 đến 10 và lưu inertia tương ứng
            for k in range(2, 11):
                kmeans = KMeans().setK(k).setSeed(42)
                model = kmeans.fit(train_data)
                inertia = model.summary.trainingCost
                inertias.append(inertia)

            # Vẽ biểu đồ Elbow
            plt.plot(range(2, 11), inertias, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            plt.show()
            # Huấn luyện mô hình trên tập huấn luyện
            kmeans = KMeans().setK(6).setSeed(42)
            model = kmeans.fit(train_data)
            # Dự đoán nhãn cụm cho tập kiểm tra
            predictions = model.transform(test_data)
            # Đánh giá hiệu suất của mô hình
            evaluator = ClusteringEvaluator()
            silhouette = evaluator.evaluate(predictions)
            print("Silhouette with squared euclidean distance = " + str(silhouette))
            # luu mo hinh
            model.write().overwrite().save("D:\\doan_AI_ungdung\\model")

        except FileNotFoundError as fe:
            print(f"File not found {fe}")
        except Exception as e:
            print(f"Error {e}")


if __name__ == "__main__":
    file_name_1 = 'data\shopping_trends.csv'
    file_name_2 = 'data\shopping_trends_updated.csv'
    data = K_mean_cluster(file_name_1, file_name_2)
    data.data_clear()


