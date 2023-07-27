from collections import Counter
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def softmax(trues, preds):
    losses = tf.math.exp(preds)
    nom = tf.math.reduce_sum(tf.multiply(trues, losses), axis=1)
    denom = tf.math.reduce_sum(losses, axis=1)
    return -tf.math.reduce_sum(tf.math.log(tf.math.divide(nom, denom)))

def balanced_loss(loss_fn, trues, preds):
    nums = Counter(trues.numpy())[1.0]
    true = []
    pred = []
    count_true = 0
    count_false = 0
    for i in np.argsort(preds.numpy())[::-1]:
        if trues[i] == 1.0:
            true.append(1.0)
            count_true += 1
        else:
            if count_false >= nums:
                continue
            true.append(0.0)
            count_false += 1
        pred.append(preds[i])
        if count_true >= nums and count_false >= nums:
            break
    pred = tf.convert_to_tensor(pred)
    return loss_fn(true, pred)

def balanced_cross_entropy(trues, preds):
    preds = tf.reshape(preds, [preds.shape[0] * preds.shape[1]])
    trues = tf.reshape(trues, [trues.shape[0] * trues.shape[1]])
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return balanced_loss(loss_fn, trues, preds)

def balanced_l2(trues, preds):
    preds = tf.reshape(preds, [preds.shape[0] * preds.shape[1]])
    trues = tf.reshape(trues, [trues.shape[0] * trues.shape[1]])
    loss_fn = tf.keras.losses.MeanSquaredError()
    return balanced_loss(loss_fn, trues, preds)

def thres(trues, preds):
    preds = tf.reshape(preds, [preds.shape[0] * preds.shape[1]])
    trues = tf.reshape(trues, [trues.shape[0] * trues.shape[1]])
    pos_loss = lambda x: max((0.9 - x, 0.0))**2
    neg_loss = lambda x: max((x - 0.5, 0.0))**2
    return tf.math.reduce_sum(tf.where(tf.equal(trues, 1.0), tf.map_fn(pos_loss, preds), tf.map_fn(neg_loss, preds)))

def balanced_thres(trues, preds):
    preds = tf.reshape(preds, [preds.shape[0] * preds.shape[1]])
    trues = tf.reshape(trues, [trues.shape[0] * trues.shape[1]])
    loss_fn = thres
    return balanced_loss(loss_fn, trues, preds)

def weighted_loss(losses, trues, weighted = True):
    if weighted:
        nums = Counter(trues.numpy())
        w_pos = 0.5 / nums[1.0]
        w_neg = 0.5 / nums[0.0]
    else:
        num = trues.shape[0]
        w_pos = 1.0 / num
        w_neg = 1.0 / num
    weights_v = tf.where(tf.equal(trues, 1.0), w_pos, w_neg)
    loss = tf.math.reduce_sum(tf.multiply(losses, weights_v))
    return loss

def weighted_cross_entropy(trues, preds, weighted = True):
    preds = tf.reshape(preds, [preds.shape[0] * preds.shape[1]])
    trues = tf.reshape(trues, [trues.shape[0] * trues.shape[1]])
    bce = K.binary_crossentropy(trues, preds, from_logits=False)
    loss = weighted_loss(bce, trues, weighted = weighted)
    return loss

def weighted_l2(trues, preds, weighted = True):
    preds = tf.reshape(preds, [preds.shape[0] * preds.shape[1]])
    trues = tf.reshape(trues, [trues.shape[0] * trues.shape[1]])
    l2 = K.square(trues-preds)
    loss = weighted_loss(l2, trues, weighted = weighted)
    return loss

def l2(trues, preds):
    preds = tf.reshape(preds, [preds.shape[0] * preds.shape[1]])
    trues = tf.reshape(trues, [trues.shape[0] * trues.shape[1]])
    loss_fn = tf.keras.losses.MeanSquaredError()
    return loss_fn(trues, preds)