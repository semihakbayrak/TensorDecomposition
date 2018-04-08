#Implementation for Logistic Tensor Factorization with TensorFlow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import tensorflow as tf

def estimated_tensor(A,R_list,sess,I,J,K):
    found = np.zeros((K,I,J))
    for k in range(K):
        f_k = sess.run(tf.sigmoid(tf.matmul(tf.matmul(A,R_list[k]),tf.transpose(A))))
        found[k,:,:] = f_k
    return found

def ltf(X,M,I,J,K,R,numberofepochs=500):
    X_M = X*M #Masked tensor
    # For every relation matrix, we keep indices that is not masked and can be used for training
    entity1_ind_list = []
    entity2_ind_list = []
    vals_list = []
    for k in range(K):
        entity1_ind = []
        entity2_ind = []
        vals = []
        for i in range(I):
            for j in range(J):
                if M[k,i,j] == 1:
                    entity1_ind.append(i)
                    entity2_ind.append(j)
                    vals.append(X[k,i,j])
        entity1_ind_list.append(entity1_ind)
        entity2_ind_list.append(entity2_ind)
        vals_list.append(vals)

    A_mat = tf.Variable(initial_value=tf.truncated_normal([I,R]), name='entities') #Initialize entity latent features
    R_mat_list = [] #List for relation latent features
    res_list = []
    res_flatten_list = []
    for k in range(K):
        R_mat_list.append(tf.Variable(initial_value=tf.truncated_normal([R,R]))) #Initalize every relation matrices
        res_list.append(tf.sigmoid(tf.matmul(tf.matmul(A_mat,R_mat_list[k]),tf.transpose(A_mat)))) #For every relation define the model which gives result
        res_flatten_list.append(tf.reshape(res_list[k],[-1])) #For every relation, reshape result from (I,I) to (I*I,1)

    # For every relation, create a result dataset which contain only not masked values.
    # For more info look at tf.gather
    estimated_list = []
    for k in range(K):
        estimated_list.append(tf.gather(res_flatten_list[k],entity1_ind_list[k]*tf.shape(res_list[k])[1]+entity2_ind_list[k]))

    # Loss function implementation part 1
    op_list = []
    base_cost_list = []
    for k in range(K):
        op_list.append(tf.add(tf.multiply(tf.log(tf.clip_by_value(estimated_list[k],1e-10,1.0)),vals_list[k]),
                              tf.multiply(tf.log(tf.clip_by_value(tf.subtract(1.0,estimated_list[k]),1e-10,1.0)),tf.subtract(1.0,vals_list[k]))))
        base_cost_list.append(tf.multiply(-1.0,tf.reduce_sum(op_list[k])))
    base_cost = tf.reduce_sum(base_cost_list)

    # regularization and loss function implementation part 2
    lda = tf.constant(.1, name='lambda')
    norm_sums = tf.add(tf.reduce_sum(tf.multiply(A_mat, A_mat)),
                       tf.reduce_sum(tf.multiply(R_mat_list, R_mat_list)))
    regularizer = tf.multiply(norm_sums, lda, 'regularizer')
    cost = tf.add(base_cost, regularizer)

    # optimizer
    lr = tf.constant(.001, name='learning_rate')
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step, 50, 0.5, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_step = optimizer.minimize(cost, global_step=global_step)

    # execute
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in xrange(numberofepochs):
        sess.run(training_step)

    X_estimated = estimated_tensor(A_mat,R_mat_list,sess,I,J,K)
    return X_estimated
