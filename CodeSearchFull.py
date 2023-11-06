import tensorflow as tf
import matplotlib.pyplot as plt
from DataReader import Reader
from DualEncoder import DualEncoderAll, create_encoder
import sys
import numpy as np
import time
import pandas as pd
from pdb import set_trace
import time
import datetime
from metrics import MAP
from MRRTesterEntireTestset import saveLinks, MRR
from getMRRScores import getMRRScores

pd.set_option('display.max_columns', None)
DATA_LOAD_PATH = ""

def learn(train_data,
          output_size = 300,
          epochs = 1000,
          validation_data=None,
          num_of_layers=1,
          dropout_rate=0.3,
          temperature=0.05,
          lr=0.001,
          decay_steps=10,
          decay_rate=0.96,
          patience=10):
    source_encoder = create_encoder(num_of_layers, input_size=train_data.element_spec['source'].shape[1], output_size=output_size, dropout_rate=dropout_rate)
    target_encoder = create_encoder(num_of_layers, input_size=train_data.element_spec['target'].shape[1], output_size=output_size, dropout_rate=dropout_rate)
    dual_encoder = DualEncoderAll(source_encoder, target_encoder, temperature = temperature)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=decay_steps, decay_rate=decay_rate,
                                                                      staircase=True)
    # dual_encoder.load("Models/")
    dual_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = dual_encoder.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[early_stopping],
    )
    fig = plt.figure()
    plt.plot(history.history["loss"], label='Train')
    plt.plot(history.history["val_loss"], label='Validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train"], loc="upper right")
    plt.legend(["validation"], loc="upper right")
    plt.savefig("Plots/" + str(datetime.datetime.now()).replace(":", "-") + ".png")
    plt.close(fig)
    return dual_encoder


def train_model(train, val, output_size=5000, batch_size=5000, epochs=1000, num_of_layers=1):
    np.random.shuffle(train.values)
    np.random.shuffle(val.values)
    print(len(train.index), len(val.index))

    td_s = train["Source"].to_list()
    td_t = train["Target"].to_list()
    source = np.array([emb for emb in td_s])
    target = np.array([emb for emb in td_t])
    tr_feature = {"source": source, "target": target}

    v_s = val["Source"].to_list()
    v_t = val["Target"].to_list()
    source = np.array([emb for emb in v_s])
    target = np.array([emb for emb in v_t])
    v_feature = {"source": source, "target": target}

    train_dataset = tf.data.Dataset.from_tensor_slices(tr_feature)
    val_dataset = tf.data.Dataset.from_tensor_slices(v_feature)

    dual_encoder = learn(train_dataset.batch(batch_size), output_size=output_size, epochs=epochs,
                         validation_data=val_dataset.batch(batch_size), num_of_layers=num_of_layers)
    return dual_encoder

def test_model(test, dual_encoder, threshold=0.5):
    ts_s = test["Source"].to_list()
    ts_t = test["Target"].to_list()
    source = np.array([emb for emb in ts_s])
    target = np.array([emb for emb in ts_t])
    ts_feature = {"source": source, "target": target}
    test_dataset = tf.data.Dataset.from_tensor_slices(ts_feature)

    start_ts = time.time()
    dual_encoder.returnEncodedTensors(ts_feature, lang="CodeSearch_PYTHON")
    mrr_score, acc, time_taken = getMRRScores(lang="CodeSearch_PYTHON")
    result = {}
    print("Test:\nTrained:")
    result["Accuracy"] = acc
    result["MRR"] = mrr_score
    result["Testing time"] = time_taken
    DATA_LOAD_PATH = "Models/"
    dual_encoder.save(DATA_LOAD_PATH)
    return result

def allLanguageExperiment(embedding="cbow",
                          output_size=2000,
                          batch_size=2000,
                          num_of_layers=1,
                          threshold=0.5):
    folder = "CodeSearch300"
    final_res = pd.DataFrame()

    r = Reader(embedding=embedding)
    r.load("CodeSearch_PYTHON_train")
    train = pd.concat([r.source_series, r.target_series], axis=1)
    np.random.shuffle(train.values)

    r = Reader(embedding=embedding)
    r.load("CodeSearch_PYTHON_valid")
    val = pd.concat([r.source_series, r.target_series], axis=1)
    np.random.shuffle(val.values)

    r = Reader(embedding=embedding)
    r.load("CodeSearch_PYTHON_test")
    test = pd.concat([r.source_series, r.target_series], axis=1)

    save_path = "Results/CodeSearchPythonFull.csv"

    print("Start")
    print("output_size", "batch_size", "thres")
    print(output_size, batch_size, threshold)
    print("*****************************")

    start_tr = time.time()
    np.random.shuffle(train.values)
    np.random.shuffle(val.values)

    dual_encoder = train_model(train, val, output_size, batch_size, num_of_layers=num_of_layers)

    end_tr = time.time()
    tr_time = round(end_tr - start_tr, 3)
    print("Training time:", tr_time)

    test_set_size = len(test.index)
    result = test_model(test, dual_encoder, threshold=0.5)

    result["Training time"] = tr_time
    df = pd.DataFrame([result])
    print("End")
    print(df)
    df.to_csv(save_path, index=False)

allLanguageExperiment(embedding="cbow",
                      output_size=2000,
                      batch_size=2000,
                      threshold=0.5,
                      num_of_layers=1)