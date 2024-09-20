import tkinter as tk
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

class CustomerClusterApp:
    def __init__(self, master):
        self.master = master
        master.title("Customer Clustering App")
        self.spark = SparkSession.builder \
            .master("local[*]") \
            .appName("CustomerClusteringApp") \
            .getOrCreate()

        self.load_model()

        self.label = tk.Label(master, text="Enter customer information")
        self.label.pack()

        self.age_label = tk.Label(master, text="Age:")
        self.age_label.pack()
        self.age_entry = tk.Entry(master)
        self.age_entry.pack()

        self.purchase_label = tk.Label(master, text="Purchase Amount (USD):")
        self.purchase_label.pack()
        self.purchase_entry = tk.Entry(master)
        self.purchase_entry.pack()

        self.rating_label = tk.Label(master, text="Review Rating:")
        self.rating_label.pack()
        self.rating_entry = tk.Entry(master)
        self.rating_entry.pack()

        self.previous_label = tk.Label(master, text="Previous Purchases:")
        self.previous_label.pack()
        self.previous_entry = tk.Entry(master)
        self.previous_entry.pack()

        self.predict_button = tk.Button(master, text="Predict Cluster", command=self.predict)
        self.predict_button.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def load_model(self):
        try:
            # Load mô hình từ thư mục đã huấn luyện trước
            model_path = "D:\\doan_AI_ungdung\\model"
            self.loaded_model = KMeansModel.load(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict(self):
        try:
            age = float(self.age_entry.get())
            purchase = float(self.purchase_entry.get())
            rating = float(self.rating_entry.get())
            previous = float(self.previous_entry.get())

            # Tạo DataFrame từ dữ liệu nhập và sử dụng VectorAssembler để tạo cột features
            features_df = self.spark.createDataFrame([(age, purchase, rating, previous)], ["Age", "Purchase", "Rating", "Previous"])
            assembler = VectorAssembler(inputCols=["Age", "Purchase", "Rating", "Previous"], outputCol="features")
            features_df = assembler.transform(features_df)

            if not hasattr(self, 'loaded_model'):
                print("Please load the model first.")
                return

            # Dự đoán nhãn cụm
            prediction = self.loaded_model.transform(features_df)
            cluster = prediction.select("prediction").first()[0]

            self.result_label.config(text=f"Predicted Cluster: {cluster}")
        except Exception as e:
            self.result_label.config(text=f"Error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CustomerClusterApp(root)
    root.mainloop()
