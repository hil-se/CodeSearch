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

pd.set_option('display.max_columns', None)
DATA_LOAD_PATH = ""

def learn(train_data,
          output_size = 300,
          epochs = 600,
          validation_data=None,
          num_of_layers=1,
          dropout_rate=0.3,
          temperature=0.05,
          lr=0.001,
          decay_steps=10,
          decay_rate=0.96,
          patience=15):
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


def train_model(train, val, output_size=5000, batch_size=5000, epochs=300, num_of_layers=1):
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
    # dual_encoder.generateBinaryClassificationData(tr_feature, data_set="train")
    # dual_encoder.generateBinaryClassificationData(v_feature, data_set="valid")
    return dual_encoder

def test_model(test, dual_encoder, threshold=0.5):
    np.random.shuffle(test.values)
    ts_s = test["Source"].to_list()
    ts_t = test["Target"].to_list()
    source = np.array([emb for emb in ts_s])
    target = np.array([emb for emb in ts_t])
    ts_feature = {"source": source, "target": target}
    test_dataset = tf.data.Dataset.from_tensor_slices(ts_feature)

    start_ts = time.time()
    m = dual_encoder.test(ts_feature, thres=threshold)
    m2 = dual_encoder.test_retrieval(ts_feature, k=3)
    m3 = dual_encoder.test_retrieval(ts_feature, k=1)
    end_ts = time.time()
    result = {}
    print("Test:")
    result["Trained"] = "Yes"
    result["Avg TL sim"] = m.cosineSimTrue()
    result["Avg FL sim"] = m.cosineSimFalse()
    result["Accuracy"] = m.accuracy()
    result["Precision"] = m.prec()
    result["Recall"] = m.tpr()
    result["F1"] = m.f1()
    result["F2"] = m.f2()
    result["MAP@3"] = m2.mavP()
    result["MAP@1"] = m3.mavP()
    result["MAA@3"] = m2.mavA()
    result["MAA@1"] = m3.mavA()
    result["MRR"] = m.mrr_score
    result["MAP"] = m.map
    result["Accuracy@3"] = m2.mAK()
    result["Accuracy@1"] = m3.mAK()
    result["Testing time"] = round(end_ts - start_ts, 3)

    start_ts = time.time()

    m = dual_encoder.testRaw(ts_feature, thres=threshold)
    m2 = dual_encoder.testRetreivalRaw(ts_feature, k=3)
    m3 = dual_encoder.testRetreivalRaw(ts_feature, k=1)

    end_ts = time.time()
    resultRaw = {}
    print("Test:")
    resultRaw["Trained"] = "No"
    resultRaw["Avg TL sim"] = m.cosineSimTrue()
    resultRaw["Avg FL sim"] = m.cosineSimFalse()
    resultRaw["Accuracy"] = m.accuracy()
    resultRaw["Precision"] = m.prec()
    resultRaw["Recall"] = m.tpr()
    resultRaw["F1"] = m.f1()
    resultRaw["F2"] = m.f2()
    resultRaw["MAP@3"] = m2.mavP()
    resultRaw["MAP@1"] = m3.mavP()
    resultRaw["MAA@3"] = m2.mavA()
    resultRaw["MAA@1"] = m3.mavA()
    resultRaw["MRR"] = m.mrr_score
    resultRaw["MAP"] = m.map
    resultRaw["Accuracy@3"] = m2.mAK()
    resultRaw["Accuracy@1"] = m3.mAK()
    resultRaw["Testing time"] = round(end_ts - start_ts, 3)

    DATA_LOAD_PATH = "Models/"
    dual_encoder.save(DATA_LOAD_PATH)
    return result, resultRaw

def allLanguageExperiment(languages,
                          embedding="cbow",
                          output_size=1000,
                          batch_size=1000,
                          test_set_size=1000,
                          num_of_layers=1,
                          threshold=0.5,
                          number_of_experiments=5):
    final_res = pd.DataFrame()
    final_res_raw = pd.DataFrame()
    for lang in languages:
        print(lang.upper())
        r = Reader(embedding)
        r.load("CodeSearch_PYTHON_train")
        train = pd.concat([r.source_series, r.target_series], axis=1)
        np.random.shuffle(train.values)
        r = Reader(embedding)
        r.load("CodeSearch_PYTHON_valid")
        val = pd.concat([r.source_series, r.target_series], axis=1)
        np.random.shuffle(val.values)
        r = Reader(embedding)
        r.load("CodeSearch_PYTHON_test")
        test_full = pd.concat([r.source_series, r.target_series], axis=1)
        save_path = "Results/CodeSearch/CodeSearch_" + lang + ".csv"

        results = []
        resultsRaw = []
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
        for i in range(number_of_experiments):
            ts_size = min(test_set_size, len(test_full.index))
            if ts_size==0:
                continue
            np.random.shuffle(test_full.values)
            test = test_full.head(ts_size)
            test_full = test_full.drop(test.index)
            # test = test_full
            test_set_size = len(test.index)
            print("Iteration -", i + 1, "/", number_of_experiments)
            result, resultRaw = test_model(test, dual_encoder, threshold=0.5)
            print(resultRaw)
            print(result)
            results.append(result)
            resultsRaw.append(resultRaw)
        df = pd.DataFrame(results)
        print("End")
        print("******************************")
        print(df)
        print("******************************")
        df.to_csv(save_path, index=False)
        df = (df.describe().loc[["mean"]])
        df = df.rename(index={"mean": lang})
        print(df)
        final_res = final_res.append(df)

        df = pd.DataFrame(resultsRaw)
        df = (df.describe().loc[["mean"]])
        df = df.rename(index={"mean": lang})
        print(df)
        final_res_raw = final_res_raw.append(df)
    print("Raw:")
    print(final_res_raw)
    print("Trained:")
    print(final_res)
    final_res.to_csv("Results/CodeSearch_PYTHON_Results.csv")
    final_res_raw.to_csv("Results/CodeSearch_PYTHON_Results_raw.csv")
    print(final_res.describe().loc[["mean"]])


languages = ["python"]
allLanguageExperiment(languages=languages,
                      embedding="cbow",
                      output_size=2000,
                      batch_size=2000,
                      test_set_size=1000,
                      threshold=0.5,
                      num_of_layers=1,
                      number_of_experiments=25)
