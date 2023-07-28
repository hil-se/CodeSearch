from pdb import set_trace
from collections import Counter
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from metrics import Metrics, Metrics_Retrieval, MAP, WebQueryMetrics, getAvgCosineScores
from loss import weighted_cross_entropy, balanced_cross_entropy, balanced_l2, weighted_l2, l2, thres, balanced_thres, softmax
import pandas as pd
import random
import math

def create_encoder(num_layers, input_size, output_size, dropout_rate):
    input = tf.keras.layers.Input(shape=(input_size,))
    x = tf.keras.layers.Dense(output_size)(input)
    pre = tf.zeros(shape=(output_size,))
    for i in range(num_layers):
        new = tf.keras.layers.Dense(output_size)(x)
        new = tf.keras.layers.Dropout(dropout_rate)(new)
        new = pre+new
        pre = x
        x = new
    output = tf.nn.relu(x)
    output = K.l2_normalize(output, axis=1)
    return tf.keras.models.Model(inputs=input, outputs=output)



class DualEncoderAll(tf.keras.Model):
    def __init__(self, source_encoder, target_encoder, temperature = 1.0, **kwargs):
        super(DualEncoderAll, self).__init__(**kwargs)
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.temperature = temperature
        self.compute_loss = self.compute_sim_loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, train_source=False, train_target=False):
        with tf.device("/gpu:0"):
            source_embeddings = self.source_encoder(features["source"], training=train_source)
        with tf.device("/gpu:1"):
            target_embeddings = self.target_encoder(features["target"], training=train_target)
        self.source_encoder.trainable = train_source
        self.target_encoder.trainable = train_target
        return source_embeddings, target_embeddings

    def compute_sim_loss(self, source_embeddings, target_embeddings):
        logits = (
                tf.matmul(source_embeddings, target_embeddings, transpose_b=True) / self.temperature
        )
        # target_similarity[i][j] is the dot_similarity(target_i, target_j).
        target_similarity = tf.matmul(
            target_embeddings, target_embeddings, transpose_b=True
        )
        # source_similarity[i][j] is the dot_similarity(source_i, source_j).
        source_similarity = tf.matmul(
            source_embeddings, source_embeddings, transpose_b=True
        )
        # targets[i][j] = avarage dot_similarity(source_i, source_j) and dot_similarity(target_i, target_j).
        targets = tf.keras.activations.softmax(
            (source_similarity + target_similarity) / (2 * self.temperature)
        )
        # Compute the loss for the sources using crossentropy
        source_loss = tf.keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )
        # Compute the loss for the targets using crossentropy
        target_loss = tf.keras.losses.categorical_crossentropy(
            y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True
        )
        # Return the mean of the loss over the batch.
        return (source_loss + target_loss) / 2

    def compute_softmax_loss(self, source_embeddings, target_embeddings):
        logits = (
                tf.matmul(source_embeddings, target_embeddings, transpose_b=True)
        )
        source_softmax = tf.keras.activations.softmax(logits, axis=1)
        source_loss = -tf.reduce_mean(tf.math.log(tf.linalg.tensor_diag_part(source_softmax)))
        target_softmax = tf.keras.activations.softmax(logits, axis=0)
        target_loss = -tf.reduce_mean(tf.math.log(tf.linalg.tensor_diag_part(target_softmax)))
        # Return the mean of the loss over the batch.
        return (source_loss + target_loss) / 2

    def train_step(self, feature, train_source=True, train_target=True):
        with tf.GradientTape() as tape:
            # Forward pass
            source_embeddings, target_embeddings = self(feature, train_source=train_source, train_target=train_target)
            loss = self.compute_loss(source_embeddings, target_embeddings)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, feature):
        source_embeddings, target_embeddings = self(feature, train_source=False, train_target=False)
        loss = self.compute_loss(source_embeddings, target_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test(self, feature, thres = 0.5):
        source_embeddings, target_embeddings = self(feature)
        simmat = tf.matmul(source_embeddings, target_embeddings, transpose_b=True)
        preds = tf.reshape(simmat, [-1])
        preds = np.array([1 if x>=thres else 0 for x in preds])
        trues = np.eye(len(source_embeddings), dtype=float)
        trues = trues.flatten()
        metric = Metrics(trues, preds)
        metric.calcCosineSim(simmat)
        #metric.mrrForSingleLinks(simmat)
        metric.mrr(source_embeddings, target_embeddings, simmat)
        matFull = metric.getCosMat(simmat, (simmat.numpy().shape[1]))
        metric.map = MAP(matFull)
        return metric

    def getEncoded(self, feature):
        source_embeddings, target_embeddings = self(feature)
        return source_embeddings, target_embeddings

    def testRaw(self, feature, thres=0.5):
        source_embeddings = feature["source"]
        target_embeddings = feature["target"]
        source_embeddings = K.l2_normalize(source_embeddings, axis=1)
        target_embeddings = K.l2_normalize(target_embeddings, axis=1)
        simmat = tf.matmul(source_embeddings, target_embeddings, transpose_b=True)
        preds = tf.reshape(simmat, [-1])
        preds = np.array([1 if x >= thres else 0 for x in preds])
        trues = np.eye(len(source_embeddings), dtype=float)
        trues = trues.flatten()
        metric = Metrics(trues, preds)
        metric.calcCosineSim(simmat)
        metric.mrr(source_embeddings, target_embeddings, simmat)
        matFull = metric.getCosMat(simmat, (simmat.numpy().shape[1]))
        metric.map = MAP(matFull)
        return metric

    def testRetreivalRaw(self, feature, k = 3):
        source_embeddings = feature["source"]
        target_embeddings = feature["target"]
        source_embeddings = K.l2_normalize(source_embeddings, axis=1)
        target_embeddings = K.l2_normalize(target_embeddings, axis=1)
        m = source_embeddings.shape[0]
        n = target_embeddings.shape[0]
        k = min((k,n))
        preds = tf.matmul(source_embeddings, target_embeddings, transpose_b=True).numpy()
        ranks = []
        for i in range(m):
            order = np.argsort(preds[i])[::-1][:k]
            ranks.append([1 if i==j else 0 for j in order])
        metric = Metrics_Retrieval(np.array(ranks))
        return metric

    def returnEncodedTensors(self, feature, lang):
        source_embeddings, target_embeddings = self(feature)
        s_es = tf.split(source_embeddings, source_embeddings.shape[0], 0)
        t_es = tf.split(target_embeddings, target_embeddings.shape[0], 0)
        results = []
        for i in range(len(s_es)):
            s_e = ((tf.squeeze(s_es[i])).numpy()).tolist()
            t_e = ((tf.squeeze(t_es[i])).numpy()).tolist()
            results.append({"Source": s_e, "Target": t_e})
        encoded_tensors = pd.DataFrame(results)
        filename = "Results/Tensors/"+lang.upper()+".csv"
        encoded_tensors.to_csv(filename, index=False)
        print(filename, "saved")
        return encoded_tensors["Source"].tolist(), encoded_tensors["Target"].tolist()

    def generateBinaryClassificationData(self, feature, data_set="train"):
        print("Generating binary classification data for", data_set, "set")
        source_embeddings, target_embeddings = self(feature)
        print("Data encoded\nGenerating links...")
        sources = []
        targets = []
        for i in range(len(source_embeddings)):
            thisTensor = source_embeddings[i]
            if isInList(sources, thisTensor) == False:
                sources.append(thisTensor)
        for i in range(len(target_embeddings)):
            thisTensor = target_embeddings[i]
            if isInList(targets, thisTensor) == False:
                targets.append(thisTensor)
        links = []
        ln = len(source_embeddings)
        for i in range(ln):
            s = source_embeddings[i]
            t = target_embeddings[i]
            s_ind = findIndex(sources, s)
            t_ind = findIndex(targets, t)
            links.append((s_ind, t_ind))
        print("Generated links\nGenerating data...")
        s_es = tf.split(source_embeddings, source_embeddings.shape[0], 0)
        t_es = tf.split(target_embeddings, target_embeddings.shape[0], 0)
        total_length = len(s_es)
        results = []
        for i in range(total_length):
            if (i % 2 == 0 and i != 0) or i == (total_length - 1):
                print("Progress:", i, "/", total_length)
            s_e = ((tf.squeeze(s_es[i])).numpy())
            t_e = ((tf.squeeze(t_es[i])).numpy())
            pos_score = np.matmul(s_e, t_e)
            results.append({"Score": pos_score, "Label": 1})
            temp = []
            while len(temp)<5:
                idx = random.randint(0, total_length-1)
                if idx!=i and idx not in temp:
                    temp.append(idx)
                    score = np.matmul(s_e, ((tf.squeeze(t_es[idx])).numpy()))
                    if (i, idx) in links:
                        results.append({"Score": score, "Label": 1})
                    else:
                        results.append({"Score": score, "Label": 0})
        print("Data generated\nSaving to file...")
        results_df = pd.DataFrame(results)
        results_df.to_csv("Results/Tensors/ClassificationData_"+data_set+".csv")
        print("Done\n\n")


    def test_retrieval(self, feature, k = 3):
        source_embeddings, target_embeddings = self(feature)
        m = source_embeddings.shape[0]
        n = target_embeddings.shape[0]
        k = min((k,n))
        preds = tf.matmul(source_embeddings, target_embeddings, transpose_b=True).numpy()
        ranks = []
        for i in range(m):
            order = np.argsort(preds[i])[::-1][:k]
            ranks.append([1 if i==j else 0 for j in order])
        metric = Metrics_Retrieval(np.array(ranks))
        return metric

    def predict(self, source, target):
        return self.source_encoder(source)*self.target_encoder(target)

    def save(self, path):
        self.source_encoder.save_weights(path+"_source")
        self.target_encoder.save_weights(path+"_target")

    def load(self, path):
        self.source_encoder.load_weights(path+"_source")
        self.target_encoder.load_weights(path+"_target")

def equalTensors(a, b):
  c = tf.math.reduce_all(tf.equal(a,b))
  return c

def isInList(l, i):
  for j in range(len(l)):
    a = l[j]
    b = equalTensors(a, i)
    if b==True:
      return True
  return False

def findIndex(l, t):
  for i in range(len(l)):
    b = equalTensors(l[i], t)
    if b==True:
      return i
  return -1


def saveLinks(data):
    source_embs = data["Source"].tolist()
    target_embs = data["Target"].tolist()
    sources = []
    targets = []
    for i in range(len(source_embs)):
        thisTensor = source_embs[i]
        if isInList(sources, thisTensor) == False:
            sources.append(thisTensor)
    for i in range(len(target_embs)):
        thisTensor = target_embs[i]
        if isInList(targets, thisTensor) == False:
            targets.append(thisTensor)
    links = []
    ln = len(source_embs)
    for i in range(ln):
        s = source_embs[i]
        t = target_embs[i]
        s_ind = findIndex(sources, s)
        t_ind = findIndex(targets, t)
        links.append((s_ind, t_ind))
    f = open("Results/Tensors/Links.txt", "w+")
    for row in links:
        f.write(str(row))
        f.write("\n")
    f.close()
    print("Saved links")


def loadLinks():
    f = open("Results/Tensors/Links.txt", "r")
    lines = f.readlines()
    links = []
    for line in lines:
        row = line.replace("(", "")
        row = row.replace(")", "")
        row = row.replace("[", "")
        row = row.replace("]", "")
        numbers = row.split(", ")
        num1 = int(numbers[0])
        num2 = int(numbers[1])
        links.append((num1, num2))
    f.close()
    return links


def getRank(listOfPairs, index, links):
    for i in range(len(listOfPairs)):
        targetIndex = listOfPairs[i][0]
        # print(index, targetIndex)
        if index == targetIndex or (index, targetIndex) in links:
            return (i + 1)
    return 0

#
# def AdvTestMetrics(data, links):
#     sources = data["Source"].tolist()
#     targets = data["Target"].tolist()
#     currentTensor = None
#     currentSims = None
#     RR = 0
#     acc = 0
#     total_length = len(sources)
#     # total_length = 50
#     start_time = time.time()
#     abs_start_time = time.time()
#     print("\nCalculating MRR score...\n")
#     for i in range(total_length):
#         if (i % 200 == 0 and i != 0) or i == (total_length - 1):
#             end_time = time.time()
#             print("Progress:", i, "/", total_length, ", Current MRR:", (RR / (i + 1)), ", Accuracy:", acc / (i + 1),
#                   ", Time since last update:", (round(end_time - start_time, 3)), "s")
#             start_time = time.time()
#         currentTensor = sources[i]
#         currentSims = []
#         for j in range(total_length):
#             currentSims.append((j, np.matmul(currentTensor, targets[j])))
#         currentSims = sorted(currentSims, key=lambda x: float(x[1]), reverse=True)
#         targetIndex = currentSims[0][0]
#         if targetIndex == i or (i, targetIndex) in links:
#             acc += 1
#         rank = getRank(currentSims, i, links)
#         if rank != 0:
#             RR += (1 / (rank))
#     mrr_score = RR / total_length
#     acc = acc / total_length
#     end_time = time.time()
#     final_time = round(end_time - abs_start_time, 3)
#     print("\nFinal MRR score:", mrr_score)
#     print("Final accuracy:", acc)
#     print("Total time taken:", final_time, "s")
#     return mrr_score, acc, final_time
