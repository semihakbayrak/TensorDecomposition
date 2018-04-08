import numpy as np
import matplotlib.pyplot as plt

def create_mask(I,J,K,rate):
    M = np.random.binomial(1,rate,I*J*K)
    M = M.reshape(K,I,J)
    return M

def read_data_from_txt(dir,I,J,K):
    lines = [line.rstrip('\n') for line in open(dir)]
    fact_list = []
    for line in lines:
        fact_list.append([int(x) for x in line.split('\t')])
    X = np.zeros((K,I,J))
    for fact in fact_list:
        k = fact[0]
        i = fact[1]
        j = fact[2]
        X[k,i,j] = 1
    return X

def read_data_from_txt_for_pf(dir,I,J,K):
    lines = [line.rstrip('\n') for line in open(dir)]
    fact_list = []
    for line in lines:
        fact_list.append([int(x) for x in line.split('\t')])
    X = 5*np.ones((K,I,J))
    for fact in fact_list:
        k = fact[0]
        i = fact[1]
        j = fact[2]
        X[k,i,j] = 1
    return X

def tpr_fpr(X,M,D,I,J,K):
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for k in range(K):
        for i in range(I):
            for j in range(J):
                if M[k,i,j] == 0:
                    if X[k,i,j] == 1:
                        if D[k,i,j] == 1:
                            tp = tp + 1
                        else:
                            fn = fn + 1
                    else:
                        if D[k,i,j] == 1:
                            fp = fp + 1
                        else:
                            tn = tn + 1
    tpr = (tp*1.0)/(tp+fn)
    fpr = (fp*1.0)/(tn+fp)
    return tpr, fpr

def tpr_fpr_train(X,M,D,I,J,K):
    tn = 0
    fn = 0
    tp = 0
    fp = 0
    for k in range(K):
        for i in range(I):
            for j in range(J):
                if M[k,i,j] == 1:
                    if X[k,i,j] == 1:
                        if D[k,i,j] == 1:
                            tp = tp + 1
                        else:
                            fn = fn + 1
                    else:
                        if D[k,i,j] == 1:
                            fp = fp + 1
                        else:
                            tn = tn + 1
    tpr = (tp*1.0)/(tp+fn)
    fpr = (fp*1.0)/(tn+fp)
    return tpr, fpr

def thresh_matrix(X_estimated,thresh,I,J,K):
    th_mat = np.zeros((K,I,J))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if X_estimated[k,i,j] > thresh:
                    th_mat[k,i,j] = 1
    return th_mat

def AUC(X,M,X_estimated,I,J,K):
    tpr_list = []
    fpr_list = []
    tpr_list_train = []
    fpr_list_train = []
    for thresh in np.linspace(0,1,51):
        D = thresh_matrix(X_estimated,thresh,I,J,K)
        tpr, fpr = tpr_fpr(X,M,D,I,J,K)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        tpr_train, fpr_train = tpr_fpr_train(X,M,D,I,J,K)
        tpr_list_train.append(tpr_train)
        fpr_list_train.append(fpr_train)
    return np.trapz(tpr_list[::-1], x=fpr_list[::-1]), np.trapz(tpr_list_train[::-1], x=fpr_list_train[::-1])

def plot_AUC_for_test(X,M,X_estimated_list,I,J,K,name_list):
    color_list = ['b-', 'y-', 'k-', 'r-', 'g-']
    list_length = len(name_list)
    for i in range(list_length):
        X_estimated = X_estimated_list[i]
        name = name_list[i]
        color = color_list[i]
        tpr_list = []
        fpr_list = []
        for thresh in np.linspace(0,1,51):
            D = thresh_matrix(X_estimated,thresh,I,J,K)
            tpr, fpr = tpr_fpr(X,M,D,I,J,K)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        print name, str(np.trapz(tpr_list[::-1], x=fpr_list[::-1]))
        plt.plot(fpr_list, tpr_list, color, label=name)
    plt.legend()
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.title('ROC Curves')
    plt.show()
