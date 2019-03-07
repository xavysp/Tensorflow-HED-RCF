
import tensorflow as tf

def sigmoid_cross_entropy_balanced(logits, label, name='cross_entrony_loss'):
    """
    Initially proposed in: 'Holistically-Nested Edge Detection (CVPR 15)'
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1.-y)
    count_pos  = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)


def class_balanced_cross_entropy_with_logits(logits,label,name='class_ballanced_cross_entropy'):

    # Initialy proposed in: 'Holistically-Nested Edge Detection (CVPR 15)'
    with tf.name_scope(name) as scope:
        logits= tf.cast(logits, tf.float32)
        label = tf.cast(label, tf.float32)

        n_positives = tf.reduce_sum(label)
        n_negatives = tf.reduce_sum(1.0-label)

        beta = n_negatives/(n_negatives+n_positives)
        pos_weight = beta / (1-beta)
        check_weight = tf.identity(beta,name='check')

        cost = tf.nn.weighted_cross_entropy_with_logits(targets=label,logits=logits,pos_weight=pos_weight)
        loss = tf.reduce_mean((1-beta)*cost)

        return tf.where(tf.equal(beta,1.0),0.0,loss)

def cross_entropy_loss_RCFpy(logits, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    # print('num pos', num_positive)
    # print('num neg', num_negative)
    # print(1.0 * num_negative / (num_positive + num_negative), 1.1 * num_positive / (num_positive + num_negative))

    cost = torch.nn.functional.binary_cross_entropy(
            logits.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost) / (num_negative + num_positive)


def cross_entropy_loss_RCFtf(logits, label,name=''):
    logits = tf.cast(logits, tf.float32)
    lab = tf.cast(label,tf.float32)
    num_positive = tf.reduce_sum(lab)
    num_negative = tf.reduce_sum(1.-lab)

    beta = num_negative/(num_positive+num_negative)

    # mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    # mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    # mask[mask == 2] = 0
    # pos_weights = beta/(1-beta)
    pos_weights = 2/(beta*(1-beta))
    a_w = label
    # all_w = tf.where(tf.equal(a_w,1.0),1.0 * num_negative / (num_positive + num_negative),
    #                  1.1 * num_negative / (num_positive + num_negative))
    # allw = tf.cond(tf.equal(a_w, 1), true_fn=1.0 * num_negative / (num_positive + num_negative),
    #                false_fn=1.1 * num_negative / (num_positive + num_negative))
    cost= tf.nn.weighted_cross_entropy_with_logits(targets=label,
                                                   logits=logits,pos_weight=pos_weights)
    loss = tf.reduce_mean((1-beta)*cost)
    # loss = tf.reduce_sum(cost)/(num_positive+num_negative)
    # tf.where(tf.equal(beta, 1.0), 0.0, loss), a_w, logits
    return tf.where(tf.equal(beta, 1.0), 0.0, loss,name=name)