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
          output_size = 1000,
          epochs = 300,
          validation_data=None,
          num_of_layers=1, # Number of layers on top of a base layer
          dropout_rate=0.3,
          temperature=0.05,
          lr=0.001, # Learning rate
          decay_steps=10,
          decay_rate=0.96,
          patience=15): # Epochs to try training despite validation loss not decreasing
    source_encoder = create_encoder(num_of_layers, input_size=train_data.element_spec['source'].shape[1], output_size=output_size, dropout_rate=dropout_rate)
    target_encoder = create_encoder(num_of_layers, input_size=train_data.element_spec['target'].shape[1], output_size=output_size, dropout_rate=dropout_rate)
    dual_encoder = DualEncoderAll(source_encoder, target_encoder, temperature = temperature)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    dual_encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = dual_encoder.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[early_stopping],
    )
    # Plotting the training and validation losses
    fig = plt.figure()
    plt.plot(history.history["loss"], label='Train')
    plt.plot(history.history["val_loss"], label='Validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train"], loc="upper right")
    plt.savefig("Plots/" + str(datetime.datetime.now()).replace(":", "-") + ".png")
    plt.close(fig)
    return dual_encoder


def train_model(train, val, output_size=5000, batch_size=5000, epochs=300, num_of_layers=1):
    # Shuffling the training and validation data
    np.random.shuffle(train.values)
    np.random.shuffle(val.values)
    print(len(train.index), len(val.index))

    # Formatting the data before feeding to the model
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

    # Training model
    dual_encoder = learn(train_dataset.batch(batch_size), output_size=output_size, epochs=epochs,
                         validation_data=val_dataset.batch(batch_size), num_of_layers=num_of_layers)
    return dual_encoder # Returns the model for testing

def test_model(test, dual_encoder, threshold=0.5):
    np.random.shuffle(test.values)  # Shuffles and formats the testing data
    ts_s = test["Source"].to_list()
    ts_t = test["Target"].to_list()
    source = np.array([emb for emb in ts_s])
    target = np.array([emb for emb in ts_t])
    ts_feature = {"source": source, "target": target}
    test_dataset = tf.data.Dataset.from_tensor_slices(ts_feature)

    start_ts = time.time()
    m = dual_encoder.test(ts_feature, thres=threshold)  # Test pairs with a fixed threshold value for Recall, Precision, F1 and F2 scores
    m2 = dual_encoder.test_retrieval(ts_feature, k=3)   # Tests for code search. For k=3, checks top 3 retrieved targets
    m3 = dual_encoder.test_retrieval(ts_feature, k=1)   # Tests for code search. For k=1, checks top retrieved target
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

def allLanguageExperiment(embedding="cbow", output_size=1000, batch_size=1000, test_set_size=1000, num_of_layers=1, number_of_experiments=5):
    threshold = 0.5 # Threshold value when creating confusion matrix
    folder = "CodeSearch300"
    #languages = ["all"]
    languages = ["python", "java", "javascript", "ruby", "go", "php"]   # Dataset divisions
    final_res = pd.DataFrame()  # Final results
    final_res_raw = pd.DataFrame()  # Evaluation of raw data
    for lang in languages:
        print(lang.upper())
        # Retrieves, shuffles and formats data files
        r = Reader(embedding+"/"+folder)
        r.load("CodeSearch_"+lang.upper()+"_train")
        train = pd.concat([r.source_series, r.target_series], axis=1)
        np.random.shuffle(train.values)
        r = Reader(embedding+"/"+folder)
        r.load("CodeSearch_" + lang.upper() + "_valid")
        val = pd.concat([r.source_series, r.target_series], axis=1)
        np.random.shuffle(val.values)
        r = Reader(embedding+"/"+folder)
        r.load("CodeSearch_" + lang.upper() + "_test")
        test_full = pd.concat([r.source_series, r.target_series], axis=1)
        save_path = "Results/CodeSearch/" + lang + "_" + str(output_size) + "_" + str(num_of_layers) + "_"+embedding+"_"+str(output_size)+".csv"

        results = []
        resultsRaw = []
        print("Start")
        print("output_size", "batch_size", "thres")
        print(output_size, batch_size, threshold)
        print("*****************************")

        start_tr = time.time()
        np.random.shuffle(train.values)
        np.random.shuffle(val.values)
        # For "Combined" experiment, train on 500,000 due to memory limitations
        if lang == "all":
            train = train.head(500000)
        dual_encoder = train_model(train, val, output_size, batch_size, num_of_layers=num_of_layers) # Returns trained model to use in testing
        end_tr = time.time()
        tr_time = round(end_tr - start_tr, 3)
        print("Training time:", tr_time)
        # Shuffles testing set and picks test_set_size number of pairs for each iteration of testing
        for i in range(number_of_experiments):
            np.random.shuffle(test_full.values)
            test = test_full.head(test_set_size)
            print("Iteration -", i + 1, "/", number_of_experiments)
            result, resultRaw = test_model(test, dual_encoder, threshold=0.5)   # result contains the evaluation on the testing set after going through the model
                                                                                # resultRaw contains evaluation on the raw testing set without passing to a model
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
    final_res.to_csv("Results/CodeSearch_Full_Results_"+str(output_size)+"_"+str(test_set_size)+".csv")
    final_res_raw.to_csv("Results/CodeSearch_Full_Results_" + str(output_size) + "_" + str(test_set_size) + "_raw.csv")


allLanguageExperiment(embedding="cbow",         # The embedding used to represent the data. Additional embeddings may need to be trained before running
                      output_size=1000,         # The dimension of the encoded vectors
                      batch_size=1000,          # Batch size
                      test_set_size=1000,       # Number of pairs to randomly pick from the test set for evaluation
                      num_of_layers=1,          # Number of layers on top of a base layer in each encoder
                      number_of_experiments=5)  # Number of times to repeat the experiment
                                                # Each experiment will have a different test set
