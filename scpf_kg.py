#Tensor extension of Sum Conditioned Poisson Factorization
#for Knowledge Graph link prediction problem

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from scipy import special

class SCPF:
    def __init__(self, X, M, I, K, R):
        self.X = X #original tensor data
        self.M = M #mask tensor. 1 for observed, 0 for unobserved
        self.I, self.J, self.K, self.R = I,I,K,R #dimensions and rank
        self.X1 = self.X*self.M #masked observations
        self.X2 = 1-self.X1 #complementary tensor

    def visualize_original(self):
        X_Show = self.X.reshape(self.K,self.I*self.J)
        plt.matshow(X_Show, cmap= cm.Greys_r)
        plt.show()

    def visualize_observation(self):
        X_Show = self.X1.reshape(self.K,self.I*self.J)
        plt.matshow(X_Show, cmap= cm.Greys_r)
        plt.show()

    #EM algorithm for Parafac decomposition
    def parafac_em(self, numberofepochs):
        eps = 1e-13 #to prevent 0 division
        N = np.ones((self.K,self.I,self.J)) #tensor of ones
        N_til = N - self.M
        #Parameter initialization
        W1 = np.random.rand(self.I,self.R)
        H1 = np.random.rand(self.J,self.R)
        G1 = np.random.rand(self.K,self.R)
        W2 = np.random.rand(self.I,self.R)
        H2 = np.random.rand(self.J,self.R)
        G2 = np.random.rand(self.K,self.R)
        for t in range(numberofepochs):
            #Update for W
            #sum_r(w*h*g)
            X1_hat = np.einsum("ir,jr,kr->kij",W1,H1,G1) + eps
            X2_hat = np.einsum("ir,jr,kr->kij",W2,H2,G2) + eps
            #sum_r_l(w*h*g)
            N_hat = X1_hat + X2_hat + eps
            #M*X/sum_r(w*h*g)
            Q_1_X = self.M*self.X1/X1_hat
            Q_2_X = self.M*self.X2/X2_hat
            #sum_k_j(M*X*h*g)/sum_r(w*h*g)
            Q_1_XHG = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",Q_1_X,G1),H1)
            Q_2_XHG = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",Q_2_X,G2),H2)
            #(1-M)N/sum_l_r(w*h*g)
            Q_N = (1-self.M)*N_til/N_hat
            #sum_k_j((1-M)*N*h*g)/sum_l_r(w*h*g)
            Q_1_NHG = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",Q_N,G1),H1)
            Q_2_NHG = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",Q_N,G2),H2)
            #tensor of ones to copy a value through dimensions
            ON = np.ones((self.K,self.I,self.J))
            #sum_j_k(h*g)
            ON_1_W = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",ON,G1),H1) + eps
            ON_2_W = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",ON,G2),H2) + eps
            #Update
            W1 = W1*(Q_1_XHG+Q_1_NHG)/ON_1_W
            W2 = W2*(Q_2_XHG+Q_2_NHG)/ON_2_W
            #Update for H
            #sum_r(w*h*g)
            X1_hat = np.einsum("ir,jr,kr->kij",W1,H1,G1) + eps
            X2_hat = np.einsum("ir,jr,kr->kij",W2,H2,G2) + eps
            #sum_r_l(w*h*g)
            N_hat = X1_hat + X2_hat + eps
            #M*X/sum_r(w*h*g)
            Q_1_X = self.M*self.X1/X1_hat
            Q_2_X = self.M*self.X2/X2_hat
            #sum_k_i(M*X*w*g)/sum_r(w*h*g)
            Q_1_XWG = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",Q_1_X,G1),W1)
            Q_2_XWG = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",Q_2_X,G2),W2)
            #(1-M)N/sum_l_r(w*h*g)
            Q_N = (1-self.M)*N_til/N_hat
            #sum_k_i((1-M)*N*w*g)/sum_l_r(w*h*g)
            Q_1_NWG = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",Q_N,G1),W1)
            Q_2_NWG = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",Q_N,G2),W2)
            #tensor of ones to copy a value through dimensions
            ON = np.ones((self.K,self.I,self.J))
            #sum_i_k(w*g)
            ON_1_H = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",ON,G1),W1) + eps
            ON_2_H = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",ON,G2),W2) + eps
            #Update
            H1 = H1*(Q_1_XWG+Q_1_NWG)/ON_1_H
            H2 = H2*(Q_2_XWG+Q_2_NWG)/ON_2_H
            #Update for G
            #sum_r(w*h*g)
            X1_hat = np.einsum("ir,jr,kr->kij",W1,H1,G1) + eps
            X2_hat = np.einsum("ir,jr,kr->kij",W2,H2,G2) + eps
            #sum_r_l(w*h*g)
            N_hat = X1_hat + X2_hat + eps
            #M*X/sum_r(w*h*g)
            Q_1_X = self.M*self.X1/X1_hat
            Q_2_X = self.M*self.X2/X2_hat
            #sum_j_i(M*X*w*h)/sum_r(w*h*g)
            Q_1_XWH = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",Q_1_X,H1),W1)
            Q_2_XWH = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",Q_2_X,H2),W2)
            #(1-M)N/sum_l_r(w*h*g)
            Q_N = (1-self.M)*N_til/N_hat
            #sum_i_j((1-M)*N*h*w)/sum_l_r(w*h*g)
            Q_1_NWH = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",Q_N,H1),W1)
            Q_2_NWH = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",Q_N,H2),W2)
            #sum_j_i(h*w)
            ON_1_G = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",ON,H1),W1) + eps
            ON_2_G = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",ON,H2),W2) + eps
            #Update
            G1 = G1*(Q_1_XWH+Q_1_NWH)/ON_1_G
            G2 = G2*(Q_2_XWH+Q_2_NWH)/ON_2_G
        #Reconstruction with estimated parameters
        X1_estimated = np.einsum("ir,jr,kr->kij",W1,H1,G1)
        X2_estimated = np.einsum("ir,jr,kr->kij",W2,H2,G2)
        X_estimated = X1_estimated/(X1_estimated+X2_estimated)
        return X_estimated, W1, H1, G1

    #Variational Bayes for Parafac decomposition
    def parafac_vi(self, numberofepochs, a_w=1., b_w=0.5, a_h=1., b_h=0.5, a_g=1., b_g=0.5):
        eps = 1e-13 #to prevent 0 division
        N = np.ones((self.K,self.I,self.J)) #tensor of ones
        N_til = N - self.M
        # initialization for exp(E[log(rv)])
        EL_W1 = np.random.rand(self.I,self.R)
        EL_H1 = np.random.rand(self.J,self.R)
        EL_G1 = np.random.rand(self.K,self.R)
        EL_W2 = np.random.rand(self.I,self.R)
        EL_H2 = np.random.rand(self.J,self.R)
        EL_G2 = np.random.rand(self.K,self.R)
        #initialization for E[rv]
        E_W1 = EL_W1
        E_W2 = EL_W2
        E_H1 = EL_H1
        E_H2 = EL_H2
        E_G1 = EL_G1
        E_G2 = EL_G2
        for t in range(numberofepochs):
            # Update for W
            X1_hat = np.einsum("ir,jr,kr->kij",EL_W1,EL_H1,EL_G1) + eps
            X2_hat = np.einsum("ir,jr,kr->kij",EL_W2,EL_H2,EL_G2) + eps
            #sum_r_l(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            N_hat = X1_hat + X2_hat + eps
            #M*X/sum_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_X = self.M*self.X1/X1_hat
            Q_2_X = self.M*self.X2/X2_hat
            #sum_k_j(M*X*exp(E[log(h)])*exp(E[log(g)]))/sum_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_XHG = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",Q_1_X,EL_G1),EL_H1)
            Q_2_XHG = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",Q_2_X,EL_G2),EL_H2)
            #(1-M)N/sum_l_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_N = (1-self.M)*N_til/N_hat
            #sum_k_j((1-M)*N*exp(E[log(h)])*exp(E[log(g)]))/sum_l_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_NHG = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",Q_N,EL_G1),EL_H1)
            Q_2_NHG = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",Q_N,EL_G2),EL_H2)
            #tensor of ones to copy a value through dimensions
            ON = np.ones((self.K,self.I,self.J))
            #sum_j_k(E[h]*E[g])
            ON_1_W = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",ON,E_G1),E_H1) + eps
            ON_2_W = np.einsum("ijr,jr->ir",np.einsum("kij,kr->ijr",ON,E_G2),E_H2) + eps
            #alfa = a + sum_j_k(E[s])
            #beta = 1/(a/b+sum_j_k[E[h]*E[g]])
            Alfa_W1 = a_w + (EL_W1*(Q_1_XHG+Q_1_NHG))
            Beta_W1 = 1/(a_w/b_w + ON_1_W)
            Alfa_W2 = a_w + (EL_W2*(Q_2_XHG+Q_2_NHG))
            Beta_W2 = 1/(a_w/b_w + ON_2_W)
            #Update
            EL_W1 = np.exp(special.digamma(Alfa_W1)) * Beta_W1
            E_W1 = Alfa_W1 * Beta_W1
            EL_W2 = np.exp(special.digamma(Alfa_W2)) * Beta_W2
            E_W2 = Alfa_W2 * Beta_W2
            #Update for H
            X1_hat = np.einsum("ir,jr,kr->kij",EL_W1,EL_H1,EL_G1) + eps
            X2_hat = np.einsum("ir,jr,kr->kij",EL_W2,EL_H2,EL_G2) + eps
            #sum_r_l(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            N_hat = X1_hat + X2_hat + eps
            #M*X/sum_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_X = self.M*self.X1/X1_hat
            Q_2_X = self.M*self.X2/X2_hat
            #sum_k_i(M*X*exp(E[log(w)])*exp(E[log(g)]))/sum_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_XWG = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",Q_1_X,EL_G1),EL_W1)
            Q_2_XWG = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",Q_2_X,EL_G2),EL_W2)
            #sum_k_i((1-M)*N*exp(E[log(w)])*exp(E[log(g)]))/sum_l_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_NWG = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",Q_N,EL_G1),EL_W1)
            Q_2_NWG = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",Q_N,EL_G2),EL_W2)
            #tensor of ones to copy a value through dimensions
            ON = np.ones((self.K,self.I,self.J))
            #sum_i_k(E[w]*E[g])
            ON_1_H = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",ON,E_G1),E_W1) + eps
            ON_2_H = np.einsum("ijr,ir->jr",np.einsum("kij,kr->ijr",ON,E_G2),E_W2) + eps
            #alfa = a + sum_i_k(E[s])
            #beta = 1/(a/b+sum_i_k[E[w]*E[g]])
            Alfa_H1 = a_h + (EL_H1*(Q_1_XWG+Q_1_NWG))
            Beta_H1 = 1/(a_h/b_h + ON_1_H)
            Alfa_H2 = a_h + (EL_H2*(Q_2_XWG+Q_2_NWG))
            Beta_H2 = 1/(a_h/b_h + ON_2_H)
            #Update
            EL_H1 = np.exp(special.digamma(Alfa_H1)) * Beta_H1
            E_H1 = Alfa_H1 * Beta_H1
            EL_H2 = np.exp(special.digamma(Alfa_H2)) * Beta_H2
            E_H2 = Alfa_H2 * Beta_H2
            #Update for G
            #sum_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            X1_hat = np.einsum("ir,jr,kr->kij",EL_W1,EL_H1,EL_G1) + eps
            X2_hat = np.einsum("ir,jr,kr->kij",EL_W2,EL_H2,EL_G2) + eps
            #sum_r_l(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            N_hat = X1_hat + X2_hat + eps
            #M*X/sum_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_X = self.M*self.X1/X1_hat
            Q_2_X = self.M*self.X2/X2_hat
            #sum_i_j(M*X*exp(E[log(w)])*exp(E[log(h)]))/sum_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_XWH = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",Q_1_X,EL_H1),EL_W1)
            Q_2_XWH = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",Q_2_X,EL_H2),EL_W2)
            #(1-M)N/sum_l_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_N = (1-self.M)*N_til/N_hat
            #sum_i_j((1-M)*N*exp(E[log(h)])*exp(E[log(w)]))/sum_l_r(exp(E[log(w)])*exp(E[log(h)])*exp(E[log(g)]))
            Q_1_NWH = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",Q_N,EL_H1),EL_W1)
            Q_2_NWH = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",Q_N,EL_H2),EL_W2)
            #tensor of ones to copy a value through dimensions
            ON = np.ones((self.K,self.I,self.J))
            #sum_j_i(E[h]*E[w])
            ON_1_G = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",ON,E_H1),E_W1) + eps
            ON_2_G = np.einsum("kir,ir->kr",np.einsum("kij,jr->kir",ON,E_H2),E_W2) + eps
            #alfa = a + sum_i_j(E[s])
            #beta = 1/(a/b+sum_i_j[E[w]*E[h]])
            Alfa_G1 = a_g + (EL_G1*(Q_1_XWH+Q_1_NWH))
            Beta_G1 = 1/(a_g/b_g + ON_1_G)
            Alfa_G2 = a_g + (EL_G2*(Q_2_XWH+Q_2_NWH))
            Beta_G2 = 1/(a_g/b_g + ON_2_G)
            #Update
            EL_G1 = np.exp(special.digamma(Alfa_G1)) * Beta_G1
            E_G1 = Alfa_G1 * Beta_G1
            EL_G2 = np.exp(special.digamma(Alfa_G2)) * Beta_G2
            E_G2 = Alfa_G2 * Beta_G2
        #Reconstruction with estimated parameters
        X1_estimated = np.einsum("ir,jr,kr->kij",E_W1,E_H1,E_G1)
        X2_estimated = np.einsum("ir,jr,kr->kij",E_W2,E_H2,E_G2)
        X_estimated = X1_estimated/(X1_estimated+X2_estimated)
        return X_estimated,E_W1,E_H1,E_G1

    #Gibbs Sampling for Parafac Decomposition
    def parafac_gibbs(self,burn_in_size, sample_size, a_w=1., b_w=0.5, a_h=1., b_h=0.5, a_g=1., b_g=0.5):
        count = 0 #to count the number of samples
        #Initially no sample for S
        S_1_sum = np.zeros((self.R,self.K,self.I,self.J))
        S_1_eval = np.zeros((self.R,self.K,self.I,self.J))
        S_2_sum = np.zeros((self.R,self.K,self.I,self.J))
        S_2_eval = np.zeros((self.R,self.K,self.I,self.J))
        #Initialization for w,h,g
        W_1 = np.random.gamma(a_w, b_w/a_w, size=(self.I,self.R))
        H_1 = np.random.gamma(a_h, b_h/a_h, size=(self.J,self.R))
        G_1 = np.random.gamma(a_g, b_g/a_g, size=(self.K,self.R))
        W_2 = np.random.gamma(a_w, b_w/a_w, size=(self.I,self.R))
        H_2 = np.random.gamma(a_h, b_h/a_h, size=(self.J,self.R))
        G_2 = np.random.gamma(a_g, b_g/a_g, size=(self.K,self.R))
        #Sampling Procedure
        for sample in range(sample_size):
            S_1, S_2 = self.sample_S_masked(self.X1,self.X2,W_1,H_1,G_1,W_2,H_2,G_2,self.M,self.I,self.J,self.K,self.R)
            W_1, H_1, G_1 = self.sample_W_H_G(S_1,W_1,H_1,G_1,a_w,b_w,a_h,b_h,a_g,b_g,self.I,self.J,self.K,self.R)
            W_2, H_2, G_2 = self.sample_W_H_G(S_2,W_2,H_2,G_2,a_w,b_w,a_h,b_h,a_g,b_g,self.I,self.J,self.K,self.R)
            if count >= burn_in_size:
                S_1_sum = S_1_sum + S_1
                S_2_sum = S_2_sum + S_2
                S_1_eval = S_1_sum/(count-burn_in_size+1)
                S_2_eval = S_2_sum/(count-burn_in_size+1)
            count = count + 1
            if count%100 == 0:
                print "Sample " +str(count)
        #Reconstruction of X with samples S
        X_estimated = np.zeros((self.K,self.I,self.J))
        for i in range(self.I):
            for j in range(self.J):
                for k in range(self.K):
                    s1 = 0
                    s2 = 0
                    for r in range(self.R):
                        s1 = s1 + S_1_eval[r,k,i,j]
                        s2 = s2 + S_2_eval[r,k,i,j]
                        X_estimated[k,i,j] = s1/(s1+s2)
        return X_estimated

    #Methods that are used in Gibbs Sampling
    @staticmethod
    def sample_S_masked(X_1,X_2,W_1,H_1,G_1,W_2,H_2,G_2,M,I,J,K,R):
        S1 = np.zeros((R,K,I,J))
        S2 = np.zeros((R,K,I,J))
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    r_r_1 = X_1[k,i,j]
                    r_r_2 = X_2[k,i,j]
                    r_n = 1
                    w_1 = W_1[i,:]
                    h_1 = H_1[j,:]
                    g_1 = G_1[k,:]
                    w_2 = W_2[i,:]
                    h_2 = H_2[j,:]
                    g_2 = G_2[k,:]
                    #Ratios for 1st tensor
                    p_r_1 = w_1*h_1*g_1
                    p_r_1_mod = p_r_1/(p_r_1.sum())
                    #Ratios for 2nd tensor
                    p_r_2 = w_2*h_2*g_2
                    p_r_2_mod = p_r_2/(p_r_2.sum())
                    if M[k,i,j] == 1:
                        #Sample from Multinomial distribution if observed
                        s1 = np.random.multinomial(r_r_1,p_r_1_mod,size=1)
                        s2 = np.random.multinomial(r_r_2,p_r_2_mod,size=1)
                    else:
                        #Sample from Multinomial distribution if unobserved
                        p_n = np.concatenate((p_r_1,p_r_2))/(p_r_1.sum()+p_r_2.sum())
                        s = np.random.multinomial(r_n,p_n,size=1)
                        s1 = s[0][0:R]
                        s2 = s[0][R:2*R]
                    S1[:,k,i,j] = s1
                    S2[:,k,i,j] = s2
        return S1, S2

    @staticmethod
    def sample_W_H_G(S,W,H,G,a_w,b_w,a_h,b_h,a_g,b_g,I,J,K,R):
        #sum_j_k(S)
        E_w = np.transpose((S.sum(1)).sum(2))
        #sum_i_k(S)
        E_h = np.transpose((S.sum(1)).sum(1))
        #sum_j_i(S)
        E_g = np.transpose((S.sum(2)).sum(2))
        #a+sum_j_k(S)
        Alfa_w = a_w + E_w
        #(a/b + sum_j_k(h*g))^-1
        Beta_w = 1.0/((a_w/b_w)+np.tile(np.sum(H*np.sum(G, axis=0),axis=0),(I,1)))
        #a+sum_i_k(S)
        Alfa_h = a_h + E_h
        #(a/b + sum_i_k(w*g))^-1
        Beta_h = 1.0/((a_h/b_h)+np.tile(np.sum(W*np.sum(G, axis=0),axis=0),(J,1)))
        #a+sum_j_i(S)
        Alfa_g = a_g + E_g
        #(a/b + sum_i_j(w*h))^-1
        Beta_g = 1.0/((a_g/b_g)+np.tile(np.sum(W*np.sum(H, axis=0),axis=0),(K,1)))
        #Sample from Gamma distribution
        Wnew = np.random.gamma(Alfa_w,Beta_w,(I,R))
        Hnew = np.random.gamma(Alfa_h,Beta_h,(J,R))
        Gnew = np.random.gamma(Alfa_g,Beta_g,(K,R))
        return Wnew,Hnew,Gnew

    #EM algorithm for Tucker decomposition
    def tucker_em(self, numberofepochs):
        eps = 1e-13 #to prevent 0 division
        N = np.ones((self.K,self.I,self.J)) #tensor of ones
        N_til = N - self.M
        # W and H initialization
        W_1 = np.random.rand(self.K,self.R,self.R)
        H_1 = np.random.rand(self.I,self.R)
        W_2 = np.random.rand(self.K,self.R,self.R)
        H_2 = np.random.rand(self.I,self.R)
        for t in range(numberofepochs):
            #Update for W
            #sum_c_d(w*h*h)
            X1_hat = np.einsum("kcd,ic,jd->kij",W_1,H_1,H_1) + eps
            X2_hat = np.einsum("kcd,ic,jd->kij",W_2,H_2,H_2) + eps
            #sum_l_c_d(w*h*h)
            N_hat = X1_hat + X2_hat
            #m*x/sum_c_d(w*h*h)
            Q_1_X = self.M*self.X1/X1_hat
            Q_2_X = self.M*self.X2/X2_hat
            #sum_j_i(m*x*h*h/sum_c_d(w*h*h))
            Q_1_XHH = np.einsum("kcj,jd->kcd",np.einsum("kij,ic->kcj",Q_1_X,H_1),H_1)
            Q_2_XHH = np.einsum("kcj,jd->kcd",np.einsum("kij,ic->kcj",Q_2_X,H_2),H_2)
            #(1-m)*N/sum_l_c_d(w*h*h)
            Q_N = (1-self.M)*N_til/N_hat
            #sum_j_i((1-m)*x*h*h/sum_l_c_d(w*h*h))
            Q_1_NHH = np.einsum("kcj,jd->kcd",np.einsum("kij,ic->kcj",Q_N,H_1),H_1)
            Q_2_NHH = np.einsum("kcj,jd->kcd",np.einsum("kij,ic->kcj",Q_N,H_2),H_2)
            #sum_i_j(h*h)
            ON_1_W = np.tile(np.einsum("ic,jd->cd",H_1,H_1),(self.K,1)).reshape(self.K,self.R,self.R) + eps
            ON_2_W = np.tile(np.einsum("ic,jd->cd",H_2,H_2),(self.K,1)).reshape(self.K,self.R,self.R) + eps
            #Update
            W_1 = W_1*(Q_1_XHH+Q_1_NHH)/ON_1_W
            W_2 = W_2*(Q_2_XHH+Q_2_NHH)/ON_2_W
            #Update for H
            #sum_c_d(w*h*h)
            X1_hat = np.einsum("kcd,ic,jd->kij",W_1,H_1,H_1) + eps
            X2_hat = np.einsum("kcd,ic,jd->kij",W_2,H_2,H_2) + eps
            #sum_l_c_d(w*h*h)
            N_hat = X1_hat + X2_hat
            #m*x/sum_c_d(w*h*h)
            Q_1_X = self.M*self.X1/X1_hat
            Q_2_X = self.M*self.X2/X2_hat
            #sum_j_k_d(m*x*w*h/sum_c_d(w*h*h))
            Q_1_XHW = np.einsum("ijcd,jd->ic",np.einsum("kij,kcd->ijcd",Q_1_X,W_1),H_1)
            Q_2_XHW = np.einsum("ijcd,jd->ic",np.einsum("kij,kcd->ijcd",Q_2_X,W_2),H_2)
            #(1-m)*N/sum_l_c_d(w*h*h)
            Q_N = (1-self.M)*N_til/N_hat
            #sum_j_k_d((1-m)*n*w*h/sum_l_c_d(w*h*h))
            Q_1_NHW = np.einsum("ijcd,jd->ic",np.einsum("kij,kcd->ijcd",Q_N,W_1),H_1)
            Q_2_NHW = np.einsum("ijcd,jd->ic",np.einsum("kij,kcd->ijcd",Q_N,W_2),H_2)
            #sum_j_k_d(w*h)
            ON_1_H = np.tile(np.einsum("kcd,jd->c",W_1,H_1),(self.I,1)).reshape(self.I,self.R) + eps
            ON_2_H = np.tile(np.einsum("kcd,jd->c",W_2,H_2),(self.I,1)).reshape(self.I,self.R) + eps
            #Update
            H_1 = H_1*(Q_1_XHW+Q_1_NHW)/ON_1_H
            H_2 = H_2*(Q_2_XHW+Q_2_NHW)/ON_2_H
        X1_estimated = np.einsum("kcd,ic,jd->kij",W_1,H_1,H_1)
        X2_estimated = np.einsum("kcd,ic,jd->kij",W_2,H_2,H_2)
        X_estimated = X1_estimated/(X1_estimated+X2_estimated)
        return X_estimated,W_1,H_1
