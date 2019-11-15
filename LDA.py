# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 20:57:52 2019

@author: Victor Biazon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

def Encoder(data):
#    y = np.asarray(data.iloc[:,-1:])
    
    enc = LabelEncoder()
    label_encoder = enc.fit(data)
    y = label_encoder.transform(data) + 1
    
    label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}
    return y, label_dict


def Covariance(data):
    Cov = np.zeros((len(data[0]),len(data[0])), dtype = float)
    for i in range(0, len(data[0])):
        for j in range(i, len(data[0])):
            sum = 0
            for k in range(0, len(data)):
                sum +=  data[k,i] * data[k,j]
            Cov[i,j] = sum/(len(data)-1)
            Cov[j,i] = Cov[i,j]
    return Cov

    

def LDA(data):
    
    #separação das variaveis independentes e dependentes
    X = np.asarray(data.iloc[:,:-1]) #separa as variaveis independentes no vetor X
    Y = np.asarray(data.iloc[:,-1:]) #separa as variaveis dependentes no vetor Y
    
    #Encoding da variavel dependente 1: 'Setosa', 2: 'Versicolor', 3:'Virginica'
    Y, Label_Dict = Encoder(Y)
    
    #calculo do GrandMean
    GMean = np.reshape(np.asarray(np.mean(data)),(1,len(np.mean(data))))
    
    #retirada da media
    M_data = np.copy(X)
    M_data = M_data - GMean


    #separando os dados por classes
    data.set_index("I", inplace=True)
    data.head()
    dataIS = data.loc['Iris-setosa']
    dataIVS = data.loc['Iris-versicolor']
    dataIVG = data.loc['Iris-virginica']

    #calculando a sample Mean
    CmeanIS = np.asarray(np.mean(dataIS))
    CmeanIVS = np.asarray(np.mean(dataIVS))
    CmeanIVG = np.asarray(np.mean(dataIVG))
    Cmean = np.vstack((CmeanIS,CmeanIVS,CmeanIVG))
    
    #retirada da media dos dados deparados por classes
    M_dataIS = np.asarray(dataIS - CmeanIS)
    M_dataIVS = np.asarray(dataIVS - CmeanIVS)
    M_dataIVG = np.asarray(dataIVG - CmeanIVG)
    
    #covariancia dos datasets
    CovIS = Covariance(M_dataIS)
    CovIVS = Covariance(M_dataIVS)
    CovIVG = Covariance(M_dataIVG)
    

    #Scatter Between
    M_mean = (Cmean - GMean).T
    Sb = np.matmul(M_mean,M_mean.T) * len(dataIS)
    
    #Scatter Within
    SwIS = (len(dataIS) - 1) * CovIS
    SwIVS = (len(dataIVS) - 1) * CovIVS
    SwIVG = (len(dataIVG) - 1) * CovIVG
    Sw = SwIS + SwIVS + SwIVG
    
    #Calculando os EigenVectors e EigenValues
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    
    #transformando os dados para o LDA em 2 dimensões.
#    Y = X * W
    W = eig_vecs[:,[0,1]]
    Y_n = X.dot(W)
    
    #separando as cores de classes
    col=[]
    for i in range(0,len(Y)):
        if Y[i]== 1:
            col.append('g') 
        elif Y[i]==2:
            col.append('r') 
        else:
            col.append('b')
    
    
    plt.figure()
    for i in range(0, len(X)):
        plt.scatter(X[i,0], X[i,1], color = col[i])
    plt.title('Data')
    plt.xlabel('X1 - SL')
    plt.ylabel('X2 - SW')
    plt.show()
    
    plt.figure()
    for i in range(0, len(X)):
        plt.scatter(Y_n[i,0], Y_n[i,1], color = col[i])
    plt.title('LDA - Data')
    plt.xlabel('LDA1')
    plt.ylabel('LDA2')
    plt.show()    
    
    return M_data, col
    
def PCA(M_data, col):
    GCov = Covariance(M_data)
    Eig_ValPCA, Eig_VecPCA = np.linalg.eig(GCov)
    PCA_Eig = Eig_VecPCA[:,:2]
    NewDataPCA = (PCA_Eig.T.dot(M_data.T)).T
    plt.figure()
    for i in range(0, len(M_data)):
        plt.scatter(NewDataPCA[i,0], NewDataPCA[i,1], color = col[i])
    M_data_pred_x = np.linspace(-8,8,20) * Eig_VecPCA[0,0] # gera as retas das PCS para plotar no grafico
    M_data_pred_y = np.linspace(-8,8,20)  * Eig_VecPCA[1,0]
    plt.plot(M_data_pred_x, M_data_pred_y, c= 'black')
    M_data_pred_x = np.linspace(-2,2,20) * Eig_VecPCA[0,1] # gera as retas das PCS para plotar no grafico
    M_data_pred_y = np.linspace(-2,2,20)  * Eig_VecPCA[1,1]
    plt.plot(M_data_pred_x, M_data_pred_y, c= 'gray')
    
    plt.title('PCA - Data')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.show()

    
    
    PCA_Eig = Eig_VecPCA[:,:3]
    NewDataPCA2 = (PCA_Eig.T.dot(M_data.T)).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, len(M_data)):
        ax.scatter(NewDataPCA2[i,0], NewDataPCA2[i,1], NewDataPCA2[i,2], zdir='z', s=20, c= col[i], depthshade=True)
    M_data_pred_x = np.linspace(-5,5,20) * Eig_VecPCA[0,0] # gera as retas das PCS para plotar no grafico
    M_data_pred_y = np.linspace(-5,5,20)  * Eig_VecPCA[1,0]
    M_data_pred_z = np.linspace(-5,5,20)  * Eig_VecPCA[2,0]
    ax.plot(M_data_pred_x, M_data_pred_y, M_data_pred_z)   
    M_data_pred_x = np.linspace(-3,3,20) * Eig_VecPCA[0,1] # gera as retas das PCS para plotar no grafico
    M_data_pred_y = np.linspace(-3,3,20)  * Eig_VecPCA[1,1]
    M_data_pred_z = np.linspace(-3,3,20)  * Eig_VecPCA[2,1]
    ax.plot(M_data_pred_x, M_data_pred_y, M_data_pred_z)  
    M_data_pred_x = np.linspace(-3,3,20) * Eig_VecPCA[0,2] # gera as retas das PCS para plotar no grafico
    M_data_pred_y = np.linspace(-3,3,20)  * Eig_VecPCA[1,2]
    M_data_pred_z = np.linspace(-3,3,20)  * Eig_VecPCA[2,2]
    ax.plot(M_data_pred_x, M_data_pred_y, M_data_pred_z)  
    
    
    return NewDataPCA, NewDataPCA2

    

data = pd.read_table('Iris.txt', decimal  = ",")

dataset = np.asarray(data)
M_data, col = LDA(data)
PCA(M_data, col)



