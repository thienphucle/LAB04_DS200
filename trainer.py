from dataloader import DataLoader
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType
from pyspark.ml.linalg import VectorUDT

class Trainer:
    def __init__(self, model, transforms):
        self.model = model
        self.sc = SparkContext(appName="FashionMNIST")
        self.ssc = StreamingContext(self.sc, 2)
        self.sqlContext = SQLContext(self.sc)
        self.dataloader = DataLoader(self.ssc, transforms)

    def train(self):
        stream = self.dataloader.parse_stream()
        stream.foreachRDD(self._train_rdd)
        self.ssc.start()
        self.ssc.awaitTermination()

    def _train_rdd(self, time, rdd):
        if not rdd.isEmpty():
            schema = StructType([
                StructField("image", VectorUDT(), True),
                StructField("label", IntegerType(), True)
            ])
            df = self.sqlContext.createDataFrame(rdd, schema)
            self.model.train(df)
            print("Batch received:", rdd.count())
