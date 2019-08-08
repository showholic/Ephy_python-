# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:58:09 2019

@author: Qixin
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from build_unit import *
from scipy.stats import zscore

#%%
pathname=r'C:\Users\Qixin\OneDrive\Lab_stuff\EPHY_Qiushou\Ctx_discrimination_data\All'
filelist=glob.glob(os.path.join(pathname,'*.pkl'))
#%%
ensemble=[]
for filename in filelist:
    ensemble.append(pd.read_pickle(filename)) 
#%% pyramidal vs interneuron 
ptdur=[unit.waveform.peak_trough_dur for unit in ensemble]
maxfr=[np.max(np.stack([unit.ctx[0]['fr'] ,unit.ctx[1]['fr']],axis=0)) for unit in ensemble]
plt.figure()
delete_ind=[]
plt.scatter(ptdur,maxfr) 
for n,unit in enumerate(ensemble):
    if (ptdur[n]<=0.4) & (maxfr[n]>=5):
        delete_ind.append(n)
for ind in delete_ind:
    del ensemble[ind]
#%%
def binned_FR(spk,dur,tbin):
    bin_edges=np.arange(0,dur,tbin)
    ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
    fr_binned=gaussian_filter1d(np.divide(np.histogram(spk,bin_edges)[0],tbin),1)
    return fr_binned,ts_bin
    
def get_binned_FR(unit,sep_trial=True):
    fr=[]
    target=[]
    context=['A','B','C']
    for c,ctx in enumerate(unit.ctx):
        for t,spk in enumerate(ctx['spkt']):
            if c==2: #context C                
                frtemp,tstemp=binned_FR(spk,60,0.5)
                fr=fr+frtemp.tolist()
                if sep_trial:
                    target=target+[context[c]+str(t+1) for i in range(len(tstemp))]
                else:
                    target=target+[context[c] for i in range(len(tstemp))]
            else:
                frtemp,tstemp=binned_FR(spk,300,0.5)
                fr=fr+frtemp.tolist()
                if sep_trial:
                    target=target+[context[c]+str(t+1) for i in range(len(tstemp))]
                else:
                    target=target+[context[c] for i in range(len(tstemp))]
    fr=np.array(fr)
    fr[np.isnan(fr)]=0
    fr=zscore(fr)    
    return fr,target

data_dict={}
for n,unit in enumerate(ensemble):
    data_dict['unit'+str(n)]=get_binned_FR(unit)[0]

data_df=pd.DataFrame(data=data_dict)
target_dict={'context':get_binned_FR(unit,sep_trial=False)[1]}
target_df2=pd.DataFrame(target_dict)
target_df=pd.DataFrame({'context':get_binned_FR(unit)[1]})
#%% LDA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#sklearn_lda=LDA(n_components=2)
#X_lda=sklearn_lda.fit_transform(data_df,target_df)
#label_dict={1:'A',2:'B',3:'C'}
#
#ax = plt.subplot(111)
#for label,marker,color in zip(
#    range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):
#
#    plt.scatter(x=X_lda[:,0][np.stack((target_df == label_dict[label]).values,axis=1)[0]],
#                y=X_lda[:,1][np.stack((target_df == label_dict[label]).values,axis=1)[0]], # flip the figure
#                marker=marker,
#                color=color,
#                alpha=0.5,
#                label=label_dict[label])
#
#plt.xlabel('LD1')
#plt.ylabel('LD2')
#
#leg = plt.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(0.5)
#
#
## hide axis ticks
#plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#        labelbottom="on", left="off", right="off", labelleft="on")
#
## remove axis spines
#ax.spines["top"].set_visible(False)  
#ax.spines["right"].set_visible(False)
#ax.spines["bottom"].set_visible(False)
#ax.spines["left"].set_visible(False)    
#
#plt.grid()
#plt.tight_layout
#plt.show()
#ctxnum=3
#label_dict={1:'A'+str(ctxnum),2:'B'+str(ctxnum),3:'C'+str((ctxnum-1)*2+1),4:'C'+str((ctxnum-1)*2+2)}
#ax = plt.subplot(111)
#for label,marker,color in zip(
#    range(1,5),('o', 'o', 's','s'),('Reds', 'Greens', 'Blues','Purples')):
#    x_coord=X_lda[:,0][np.stack((target_df2 == label_dict[label]).values,axis=1)[0]]
#    s=np.linspace(0,1,num=len(x_coord)).tolist()
#    s.reverse()
#    plt.scatter(x=X_lda[:,0][np.stack((target_df2 == label_dict[label]).values,axis=1)[0]],
#                y=X_lda[:,1][np.stack((target_df2 == label_dict[label]).values,axis=1)[0]], # flip the figure
#                marker=marker,
#                c=s,
#                s=8,
#                cmap=color,
#                alpha=0.5,
#                label=label_dict[label])
#
#plt.xlabel('LD1')
#plt.ylabel('LD2')
#
#leg = plt.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(0.5)
#
#
## hide axis ticks
#plt.tick_params(axis="both", which="both", bottom="off", top="off",  
#        labelbottom="on", left="off", right="off", labelleft="on")
#
## remove axis spines
#ax.spines["top"].set_visible(False)  
#ax.spines["right"].set_visible(False)
#ax.spines["bottom"].set_visible(False)
#ax.spines["left"].set_visible(False)    
#
#plt.grid()
#plt.tight_layout
#plt.show()

#%% PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
principalComponents=pca.fit_transform(data_df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
#%% plot PCA1
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ctx_id=np.unique(target_df['context'])
a, b, c=[plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]
ai,bi,ci=[1,1,1]
colors=[]
for ctxname in ctx_id:
    if ctxname[0]=='A':
        colors.append(a(ai))
        ai-=0.1
    elif ctxname[0]=='B':
        colors.append(b(bi))
        bi-=0.1
    elif ctxname[0]=='C':
        colors.append(c(ci))
        ci-=0.1 

for target,color in zip(ctx_id,colors):
    ind2keep=target_df['context']==target
    ax.scatter(principalDf.loc[ind2keep,'principal component 1'],
               principalDf.loc[ind2keep,'principal component 2'],
               c=color,
               s=50)
#%%
