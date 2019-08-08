# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:19:44 2019

@author: Qixin
"""
#%%
feature_dict={i:label for i,label in zip(
        range(4),
        ('sepal length in cm',
         'sepal width in cm',
         'petal length in cm',
         'petal width in cm',
                ))}
#%% Read dataset
import pandas as pd
df=pd.io.parsers.read_csv(
        filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None,
        sep=',')
df.columns=[l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how='all',inplace=True)
#%%
from sklearn.preprocessing import LabelEncoder
X=df.iloc[:,0:4].values
y=df['class label'].values
enc=LabelEncoder()
label_encoder = enc.fit(y)
y=label_encoder.transform(y)+1
label_dict={1:'Setosa',2:'Versicolor',3:'Virginica'}
#%%
from matplotlib import pyplot as plt
import numpy as np
import math
fig,axes=plt.subplots(2,2,figsize=(12,6))
for ax,cnt in zip(axes.ravel(),range(4)):
    min_b=math.floor(np.min(X[:,cnt]))
    max_b=math.ceil(np.max(X[:,cnt]))
    bins=np.linspace(min_b,max_b,25)
    for lab,col in zip(range(1,4), ('blue', 'red', 'green')):
        ax.hist(X[y==lab,cnt],
                color=col,
                label='class %s' %label_dict[lab],
                bins=bins,
                alpha=0.5)
    ylims=ax.get_ylim()
    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims)+2])
    ax.set_xlabel(feature_dict[cnt])
    ax.set_title('Iris histogram #%s' %str(cnt+1))

    # hide axis ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')

fig.tight_layout()       

plt.show()
#%% Step1: computing the d-dimensional mean vectors 
np.set_printoptions(precision=4)
mean_vectors=[]
for cl in range(1,4):
    mean_vectors.append(np.mean(X[y==cl],axis=0))

#%% Step2: Computing the scatter matrices 
#2.1 Within-class scatter matrix Sw
S_W=np.zeros((4,4))
for cl,mv in zip(range(1,4),mean_vectors):
    class_sc_mat=np.zeros((4,4))
    for row in X[y==cl]:
        row,mv = row.reshape(4,1), mv.reshape(4,1)
        class_sc_mat+=(row-mv).dot((row-mv).T)
    S_W += class_sc_mat
#2.2 Between-class scatter matrix SB
overall_mean=np.mean(X,axis=0)
S_B=np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):
    n=X[y==i+1,:].shape[0]
    mean_vec=mean_vec.reshape(4,1)
    overall_mean=overall_mean.reshape(4,1)
    S_B+= n*(mean_vec-overall_mean).dot((mean_vec-overall_mean).T)
#%% Solve eigenvalue 
eig_vals,eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
#%% Sort eigenvector by decreasing eigenvalues
eig_pairs=[(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs=sorted(eig_pairs,key=lambda k:k[0], reverse=True) 
#%% W: Choose k eigenvectors with the largest eigenvalues 
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
#%% Transform the samples onto the new wubspace
X_lda=X.dot(W)

