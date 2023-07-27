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
    dual_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = dual_encoder.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[early_stopping]
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
    np.random.shuffle(test.values)
    ts_s = test["Source"].to_list()
    ts_t = test["Target"].to_list()
    source = np.array([emb for emb in ts_s])
    target = np.array([emb for emb in ts_t])
    ts_feature = {"source": source, "target": target}
    test_dataset = tf.data.Dataset.from_tensor_slices(ts_feature)

    dual_encoder.returnEncodedTensors(ts_feature, lang="DGMS")
    saveLinks(lang="DGMS")
    mrr_score, acc, time_taken = MRR(lang="DGMS", duplicates=True)
    start_ts = time.time()
    result = {}
    print("Test:")
    result["Accuracy"] = acc
    result["MRR"] = mrr_score
    print(result)
    m = dual_encoder.test(ts_feature, thres=threshold)
    m3 = dual_encoder.test_retrieval(ts_feature, k=1)
    end_ts = time.time()
    fin_time = round(end_ts - start_ts, 3)
    result["Testing time"] = fin_time
    result["MAP"] = m.map
    result["MAP@1"] = m3.mavP()
    result["MAA@1"] = m3.mavA()
    result["Accuracy@1"] = m3.mAK()
    DATA_LOAD_PATH = "Models/"
    dual_encoder.save(DATA_LOAD_PATH)
    return result

def allLanguageExperiment(embedding="cbow",
                          output_size=2000,
                          batch_size=2000,
                          num_of_layers=1,
                          threshold=0.5,
                          num_of_experiments=10):
    folder = "CodeSearch300"
    final_res = pd.DataFrame()
    final_res_raw = pd.DataFrame()

    r = Reader(embedding)
    r.load("DGMS")

    save_path = "Results/DGMS.csv"

    print("Start")
    print("output_size", "batch_size", "thres")
    print(output_size, batch_size, threshold)
    print("*****************************")
    results = []
    for i in range(num_of_experiments):
        full_data = pd.concat([r.source_series, r.target_series], axis=1)
        np.random.shuffle(full_data.values)
        train = full_data.head(int((len(full_data.index)*0.8)))
        val = full_data.drop(train.index)
        test = val.head(1000)
        val = val.drop(test.index)

        print(i)
        start_tr = time.time()
        dual_encoder = train_model(train, val, output_size, batch_size, num_of_layers=num_of_layers)
        end_tr = time.time()
        tr_time = round(end_tr - start_tr, 3)
        print("Training time:", tr_time)
        result = test_model(test, dual_encoder, threshold=0.5)
        result["Training time"] = tr_time
        results.append(result)
    df = pd.DataFrame(results)
    print("End")
    print(df)
    df.to_csv(save_path, index=False)

allLanguageExperiment(embedding="cbow",
                      output_size=2000,
                      batch_size=2000,
                      threshold=0.5,
                      num_of_layers=1,
                      num_of_experiments=5)
