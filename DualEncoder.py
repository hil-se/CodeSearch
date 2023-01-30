from pdb import set_trace
from collections import Counter
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
from metrics import Metrics, Metrics_Retrieval, MAP
from loss import weighted_cross_entropy, balanced_cross_entropy, balanced_l2, weighted_l2, l2, thres, balanced_thres, softmax
import pandas as pd

def create_encoder(num_layers, input_size, output_size, dropout_rate):
    input = tf.keras.layers.Input(shape=(input_size,))
    x = tf.keras.layers.Dense(output_size)(input)
    pre = tf.zeros(shape=(output_size,))
    # Additional layers
    # num_layers here refers to num_of_layers from CodeSearch.py
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
        # metric.mrrForSingleLinks(simmat)
        metric.mrr(source_embeddings, target_embeddings, simmat)
        matFull = metric.getCosMat(simmat, (simmat.numpy().shape[1]))
        metric.map = MAP(matFull)
        return metric

    def getEncoded(self, feature):
        # Get model-encoded vectors for given data
        source_embeddings, target_embeddings = self(feature)
        return source_embeddings, target_embeddings

    def testRaw(self, feature, thres=0.5):
        # Evaluate raw data not encoded by the model
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
        return metric

    def testRetreivalRaw(self, feature, k = 3):
        # Evaluate raw data not encoded by the model for retrieval task
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

    def returnEncodedTensors(self, feature):
        # Writes encoded vectors to file, and returns them as lists
        source_embeddings, target_embeddings = self(feature)
        s_es = tf.split(source_embeddings, source_embeddings.shape[0], 0)
        t_es = tf.split(target_embeddings, target_embeddings.shape[0], 0)
        encoded_tensors = pd.DataFrame(columns=["Source", "Target"])
        for i in range(len(s_es)):
            s_e = (tf.squeeze(s_es[i])).numpy()
            t_e = (tf.squeeze(t_es[i])).numpy()
            encoded_tensors = encoded_tensors.append({"Source": s_e, "Target": t_e}, ignore_index=True)
        encoded_tensors.to_csv("Results/Tensors/CodeSearch_Tensors.csv", index=False)
        return encoded_tensors["Source"].tolist(), encoded_tensors["Target"].tolist()

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


