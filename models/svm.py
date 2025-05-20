import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyspark.sql.dataframe import DataFrame

class SVM:
    def __init__(self):
        self.model = LinearSVC()

    def train(self, df: DataFrame):
        X = np.array(df.select("image").collect()).reshape(-1, 784)
        y = np.array(df.select("label").collect()).reshape(-1)
        self.model.fit(X, y)
        preds = self.model.predict(X)
        return preds.tolist(), accuracy_score(y, preds), precision_score(y, preds, average='macro'), recall_score(y, preds, average='macro'), f1_score(y, preds, average='macro')
