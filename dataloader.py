import numpy as np, json
from pyspark.ml.linalg import DenseVector
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream

class DataLoader:
    def __init__(self, context, transforms):
        self.context = context
        self.stream = self.context.socketTextStream("localhost", 6100)
        self.transforms = transforms

    def parse_stream(self) -> DStream:
        stream = self.stream.map(lambda line: json.loads(line))
        stream = stream.flatMap(lambda x: x.values())
        stream = stream.map(lambda x: [np.array([x[f"feature-{i}"] for i in range(784)]).reshape(28, 28), x["label"]])
        stream = stream.map(lambda x: [self.transforms.transform(x[0]).reshape(-1).tolist(), x[1]])
        stream = stream.map(lambda x: [DenseVector(x[0]), x[1]])
        return stream
