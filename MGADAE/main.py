import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import os
import random
import time
import h5py
from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
from scipy.io import loadmat

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    #print('adj')
    #print(adj.shape)
    #print(adj)  构建网络
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()
    X = constructNet(train_drug_dis_matrix)
    #print('x')
    #print(X.shape)
    #print(X)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_dis_matrix.shape[0], name='MGADAE')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_dis_matrix.shape[0], num_v=train_drug_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            # start_time  = time.time()
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            # end_time = time.time()
            # epoch_time  = end_time - start_time
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res


def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))
    
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()

    # print(index_matrix)
    random.seed(seed)
    random.shuffle(random_index)
    
    '''
    index_matrix_neg = np.mat(np.where(drug_dis_matrix == 0))
    association_nam_neg = index_matrix_neg.shape[1]
    random_index_eng = index_matrix_neg.T.tolist()
    random.shuffle(random_index_neg)
    '''
    
    k_folds = 5
    row_idx = None
    col_idx = not None
    interactions = drug_dis_matrix  # 关联矩阵
    drug_len = drug_dis_matrix.shape[0]   # drug的个数
    dis_len = drug_dis_matrix.shape[1]   # miRNA的个数
    
    metric = np.zeros((1, 7))  # 7个指标
    if k_folds == -1:
        if row_idx is not None:
            row_idxs = list(range(interactions.shape[0])) if k_folds==-1 else [row_idx]
            for idx in row_idxs:
                #if idx > 1:
                #    continue
                #print(idx)
                print("------this is %dth cross validation of %d------" % (idx,drug_len))
                train_matrix = get_fold_local_mask(interactions, row_idx=idx)
                
                #print(np.count_nonzero(train_matrix))
                drug_disease_res = PredictScore(
                    train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
                predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)
                metric_tmp = cv_model_evaluate(drug_dis_matrix, predict_y_proba, train_matrix)
                print(metric_tmp)
                metric += metric_tmp
                del train_matrix
                gc.collect()
            print(metric / drug_len)
            metric = np.array(metric / drug_len)
               
                
        elif col_idx is not None:
            col_idxs = list(range(interactions.shape[1])) if k_folds==-1 else [col_idx]
            for idx in col_idxs:
                #if idx > 1:
                #    continue
                #print(idx)
                print("------this is %dth cross validation of %d------" % (idx,dis_len))
                train_matrix= get_fold_local_mask(interactions, col_idx=idx)
                #print(np.count_nonzero(train_matrix))
                drug_disease_res = PredictScore(
                    train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
                predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)
                #metric_tmp = cv_model_evaluate(drug_dis_matrix, predict_y_proba, train_matrix)
                #print(metric_tmp)
                test_ma = interactions[:,idx]
                predict_y = predict_y_proba[:,idx]
                metric_tmp = cv_model_evaluate(test_ma, predict_y, train_matrix)
                
                log_dir='../result-predict/'
                np.savetxt(os.path.join(log_dir,'id_score_{}.csv'.format(idx)), predict_y, delimiter = ',')
                np.savetxt(os.path.join(log_dir,'id_label_{}.csv'.format(idx)), test_ma, delimiter = ',')
                
                print(metric_tmp)
                metric += metric_tmp
                del train_matrix
                gc.collect()
                
            print(metric / dis_len)
            metric = np.array(metric / dis_len)
        
        
                    
    else:
        #---抽取负样本
        neg = ([], [])  # 存储负样本的坐标
        neg_nam = association_nam
        while len(neg[0]) < neg_nam:
            i, j = random.randint(0, drug_len-1), random.randint(0, dis_len-1)
            if drug_dis_matrix[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        neg_row = np.array(neg[0])
        neg_col = np.array(neg[1])
        neg_index = np.stack([neg_row, neg_col])
        random_neg_index = neg_index.T.tolist()
        #print(neg_index)
        #print(neg_index.shape)
        CV_size = int(association_nam / k_folds)
        CV_neg_size = int(neg_nam / k_folds)
        temp = np.array(random_index[:association_nam - association_nam %
                                     k_folds]).reshape(k_folds, CV_size,  -1).tolist()
        temp[k_folds - 1] = temp[k_folds - 1] + \
            random_index[association_nam - association_nam % k_folds:]
        random_index = temp
        
        temp_neg = np.array(random_neg_index[:neg_nam - neg_nam %
                                     k_folds]).reshape(k_folds, CV_neg_size,  -1).tolist()
        temp_neg[k_folds - 1] = temp_neg[k_folds - 1] + \
            random_neg_index[neg_nam - neg_nam % k_folds:]
        random_neg_index = temp_neg
        
        
        
        #kfold = KFold(n_splits=n_splits, shuffle=True, random_state=666)
        
        '''
        CV_size = int(association_nam / k_folds)
        temp = np.array(random_index[:association_nam - association_nam %
                                     k_folds]).reshape(k_folds, CV_size,  -1).tolist()
        temp[k_folds - 1] = temp[k_folds - 1] + \
            random_index[association_nam - association_nam % k_folds:]
        random_index = temp
        '''
        
        
        print("seed=%d, evaluating drug-disease...." % (seed))
        
        for k in range(k_folds):
        #k=0
        #for (train_neg_idx, test_neg_idx) in kfold.split(neg_row):
            #if k!=0:
            #    continue
           
            pos_index = random_index[k]
            neg_index = random_neg_index[k]
            print("------this is %dth cross validation------" % (k+1))
            train_matrix = np.matrix(drug_dis_matrix, copy=True)
            test_matrix = train_matrix[tuple(np.array(random_index[k]).T)]
            train_matrix[tuple(np.array(random_index[k]).T)] = 0
            #print(train_matrix)
            #print(train_matrix.shape)
            drug_disease_res = PredictScore(
                train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp)
            predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)
            predict_y = predict_y_proba[tuple(np.array(random_index[k]).T)]
            #找到test中的pos
            #print(test_matrix.shape)
            #print(test_predict_y.shape)
            metric_tmp = cv_model_evaluate(drug_dis_matrix, predict_y_proba, neg_index, pos_index,k)
            print(metric_tmp)
            metric += metric_tmp
            del train_matrix
            del test_matrix
            gc.collect()
            
        print(metric / k_folds)
        metric = np.array(metric / k_folds)
    
    return metric

def get_fold_local_mask( interactions, row_idx=None, col_idx=None):
    #train_mask = np.ones_like(interactions, dtype="bool")
    #test_mask = np.zeros_like(interactions, dtype="bool")
    #train_matrix = np.zeros_like(interactions)
    #test_matrix = np.zeros_like(interactions, dtype="bool")
    if row_idx is not None:
        train_matrix = np.matrix(interactions, copy=True)
        train_matrix[row_idx, :] = 0
        #test_mask[np.ones(interactions.shape[1], dtype="int")*row_idx,np.arange(interactions.shape[1])] = True
    elif col_idx is not None:
        train_matrix = np.matrix(interactions, copy=True)
        train_matrix[:,col_idx]= 0
        #test_mask[np.arange(interactions.shape[0]),np.ones(interactions.shape[0], dtype="int") * col_idx] = True
        
    return train_matrix


if __name__ == "__main__":
     
    ##############   D1 DATASET   ##################
    mir_mkl = np.loadtxt('./D1data/m_mkl.csv', delimiter=',')
    dis_mkl = np.loadtxt('./D1data/d_mkl.csv', delimiter=',')
    mir_dis_matrix = np.loadtxt('./D1data/mda.csv', delimiter=',')
    

    epoch = 4000
    emb_dim = 128


    lr = 0.01
    adjdp = 0.6
    dp = 0.4
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    for i in range(circle_time):
        result += cross_validation_experiment(mir_dis_matrix, mir_mkl, dis_mkl, 1, epoch, emb_dim, dp, lr, adjdp)
    average_result = result / circle_time
    print(average_result)
