import pandas as pd
import numpy as np
import tensorflow as tf
from pdb import set_trace

class Reader:
    def __init__(self, embedding="", testset=False):
        if testset:
            self.path = "Results/Tensors/"
        else:
            if embedding=="":
                self.path = "Data/"
            else:
                self.path = "Data/"+embedding+"/CodeSearch300/"

    def load(self, name):
        self.source, self.target= self.load_embeddings(self.path + name + ".csv")
        self.feature = {"source": self.source, "target": self.target}
        self.size = len(self.source)
        return self.feature


    def load_embeddings(self, file):
        def parse(emb_str):
            emb_str = str(emb_str)
            emb_str = emb_str.replace(",", "")
            emb_list = emb_str[1:-1].split()
            embeddings = [float(emb.strip()) for emb in emb_list]
            return embeddings
        df = pd.read_csv(file)
        df = df.dropna()
        embeddings = df["Source"].apply(parse)
        self.source_series = embeddings
        source = np.array([emb for emb in embeddings])
        embeddings = df["Target"].apply(parse)
        self.target_series = embeddings
        target = np.array([emb for emb in embeddings])
        return source, target

