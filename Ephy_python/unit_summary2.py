# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 16:28:35 2019

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
    if (ptdur[n]<=0.4) or (maxfr[n]>=10):
        delete_ind.append(n)
ensemble=np.delete(ensemble,delete_ind).tolist()
#%%
def pie_chart_summary_ranksum(ensemble):
    num_place_cell=0
    num_ctx_cell=0
    num_non_ctx_cell=0
    num_ctx_and_place_cell=0
    num_A=0
    num_B=0
    num_AP=0    
    num_BP=0
    num_OP=0
    num_all=len(filelist)
    for unit in ensemble:
        CS=ctx_ranksum(unit)
        if unit.placefield=='A' or unit.placefield=='B' or unit.placefield=='Both':
            num_place_cell+=1
            if CS!='O':
                num_ctx_and_place_cell+=1
        if CS!= 'O':
            #print(unit.context_selectivity[0])
            num_ctx_cell+=1
            if (CS.upper())=='A':
                num_A+=1
                if unit.placefield=='A' or unit.placefield=='B' or unit.placefield=='Both':
                    num_AP+=1
            if (CS.upper())=='B':
                num_B+=1
                if unit.placefield=='A' or unit.placefield=='B' or unit.placefield=='Both':
                    num_BP+=1
        else:
            num_non_ctx_cell+=1
            if unit.placefield=='A' or unit.placefield=='B' or unit.placefield=='Both':
                num_OP+=1    
    group_names=['A','B','others']
    group_size=[num_A,num_B,num_non_ctx_cell]
    subgroup_names=['',' ','',' ','',' ']
    subgroup_size=[num_AP,num_A-num_AP,num_BP,num_B-num_BP,num_OP,num_non_ctx_cell-num_OP]
    a, b, c, d, e=[plt.cm.Reds, plt.cm.Blues, plt.cm.Greens,plt.cm.Oranges,plt.cm.gray]
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(group_size, radius=1.3, labels=['{:.1%}'.format(num_A/num_all),
                                                      '{:.1%}'.format(num_B/num_all),
                                                      '{:.1%}'.format(num_non_ctx_cell/num_all)], 
        colors=[a(0.6), b(0.6), c(0.6)],labeldistance=1.05)
    plt.setp( mypie, width=0.3, edgecolor='white')
    
    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.4, colors=[d(0.6), e(0.6), d(0.6), e(0.6),d(0.6), e(0.6)])
    plt.setp( mypie2, width=0.4, edgecolor='white')
    plt.margins(0,0)
    A_patch=mpatches.Patch(color=a(0.6),label='Context A cells')
    B_patch=mpatches.Patch(color=b(0.6),label='Context B cells')
    O_patch=mpatches.Patch(color=c(0.6),label='Non-context cells')
    orange_patch=mpatches.Patch(color=d(0.6),label='Place cells')
    grey_patch=mpatches.Patch(color=e(0.6),label='Non-place cells')
    plt.legend(handles=[A_patch,B_patch,O_patch,orange_patch,grey_patch],loc='upper right',bbox_to_anchor=(1.15, 1.1))
    plt.title('Summary of ' + str(num_all) + ' units',y=1.1)
    # show it
    plt.show()
#%% control locomotion 
cdi=[unit.cdi[0] for unit in ensemble]
plt.figure(dpi=150)
loc_pref=[]
for unit in ensemble:
    ctx=unit.split_context(unit.pos)
    td1=ctx[0]['travel_distance']
    td2=ctx[1]['travel_distance']
    loc_pref.append((td1-td2)/(td1+td2))
plt.scatter(loc_pref,cdi)
plt.xlim([-0.5,0.5])
plt.ylim([-1,1])
plt.xlabel('Locomotion selectivity')
plt.ylabel('Context selectivity')
plt.axvline(x=0,linestyle='dashed')
plt.axhline(y=0,linestyle='dashed')
corr=np.corrcoef(loc_pref,cdi)[0][1]
plt.title('r = %.3f' %corr)

#%%
def pie_chart_summary_strict_bootstrap(ensemble):
    num_place_cell=0
    num_ctx_cell=0
    num_non_ctx_cell=0
    num_ctx_and_place_cell=0
    num_A=0
    num_B=0
    num_AP=0    
    num_BP=0
    num_OP=0
    num_all=len(filelist)
    for unit in ensemble:
        CS=bootstrap_context(unit,eval_method='fr1')[0]
        if unit.placefield=='A' or unit.placefield=='B' or unit.placefield=='Both':
            num_place_cell+=1
            if CS!='others':
                num_ctx_and_place_cell+=1
        if CS!= 'others':
            #print(unit.context_selectivity[0])
            num_ctx_cell+=1
            if (CS.upper())=='A':
                num_A+=1
                if unit.placefield=='A' or unit.placefield=='B' or unit.placefield=='Both':
                    num_AP+=1
            if (CS.upper())=='B':
                num_B+=1
                if unit.placefield=='A' or unit.placefield=='B' or unit.placefield=='Both':
                    num_BP+=1
        else:
            num_non_ctx_cell+=1
            if unit.placefield=='A' or unit.placefield=='B' or unit.placefield=='Both':
                num_OP+=1    
    group_names=['A','B','others']
    group_size=[num_A,num_B,num_non_ctx_cell]
    subgroup_names=['',' ','',' ','',' ']
    subgroup_size=[num_AP,num_A-num_AP,num_BP,num_B-num_BP,num_OP,num_non_ctx_cell-num_OP]
    a, b, c, d, e=[plt.cm.Reds, plt.cm.Blues, plt.cm.Greens,plt.cm.Oranges,plt.cm.gray]
    fig, ax = plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(group_size, radius=1.3, labels=['{:.1%}'.format(num_A/num_all),
                                                      '{:.1%}'.format(num_B/num_all),
                                                      '{:.1%}'.format(num_non_ctx_cell/num_all)], 
        colors=[a(0.6), b(0.6), c(0.6)],labeldistance=1.05)
    plt.setp( mypie, width=0.3, edgecolor='white')
    
    # Second Ring (Inside)
    mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.4, colors=[d(0.6), e(0.6), d(0.6), e(0.6),d(0.6), e(0.6)])
    plt.setp( mypie2, width=0.4, edgecolor='white')
    plt.margins(0,0)
    A_patch=mpatches.Patch(color=a(0.6),label='Context A cells')
    B_patch=mpatches.Patch(color=b(0.6),label='Context B cells')
    O_patch=mpatches.Patch(color=c(0.6),label='Non-context cells')
    orange_patch=mpatches.Patch(color=d(0.6),label='Place cells')
    grey_patch=mpatches.Patch(color=e(0.6),label='Non-place cells')
    plt.legend(handles=[A_patch,B_patch,O_patch,orange_patch,grey_patch],loc='upper right',bbox_to_anchor=(1.15, 1.1))
    plt.title('Summary of ' + str(num_all) + ' units',y=1.1)
    # show it
    plt.show()
#%% control locomotion 
cdi=[unit.cdi[0] for unit in ensemble]
plt.figure(dpi=150)
loc_pref=[]
for unit in ensemble:
    ctx=unit.split_context(unit.pos)
    td1=ctx[0]['travel_distance']
    td2=ctx[1]['travel_distance']
    loc_pref.append((td1-td2)/(td1+td2))
plt.scatter(loc_pref,cdi)
plt.xlim([-0.5,0.5])
plt.ylim([-1,1])
plt.xlabel('Locomotion selectivity')
plt.ylabel('Context selectivity')
plt.axvline(x=0,linestyle='dashed')
plt.axhline(y=0,linestyle='dashed')
corr=np.corrcoef(loc_pref,cdi)[0][1]
plt.title('r = %.3f' %corr)
#%%
def binned_FR(spk,dur,tbin):
    bin_edges=np.arange(0,dur,tbin)
    ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
    fr_binned=gaussian_filter1d(np.divide(np.histogram(spk,bin_edges)[0],tbin),1)
    return fr_binned,ts_bin
    
def get_binned_FR(unit):
    fr=[]
    target=[]
    context=['A','B','C']
    for c,ctx in enumerate(unit.ctx):
        for t,spk in enumerate(ctx['spkt']):
            if c==2: #context C                
                frtemp,tstemp=binned_FR(spk,60,0.5)
                fr=fr+frtemp.tolist()
                target=target+[context[c]+str(t+1) for i in range(len(tstemp))]
            else:
                frtemp,tstemp=binned_FR(spk,300,0.5)
                fr=fr+frtemp.tolist()
                target=target+[context[c]+str(t+1) for i in range(len(tstemp))]
    fr=np.array(fr)
    fr[np.isnan(fr)]=0
    fr=zscore(fr)
    return fr,target

data_dict={}
for n,unit in enumerate(ensemble):
    data_dict['unit'+str(n)]=get_binned_FR(unit)[0]

data_df=pd.DataFrame(data=data_dict)
target_dict={'context':get_binned_FR(unit)[1]}
target_df=pd.DataFrame(target_dict)

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
#%% LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
sklearn_lda=LDA(n_components=2)
X_lda=sklearn_lda.fit_transform(data_df,target_df)
#%%
ctxnum=2
label_dict={1:'A'+str(ctxnum),2:'B'+str(ctxnum),3:'C'+str(ctxnum)}

ax = plt.subplot(111)
for label,marker,color in zip(
    range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

    plt.scatter(x=X_lda[:,0][np.stack((target_df == label_dict[label]).values,axis=1)[0]],
                y=X_lda[:,1][np.stack((target_df == label_dict[label]).values,axis=1)[0]], # flip the figure
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label])

plt.xlabel('LD1')
plt.ylabel('LD2')

leg = plt.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.5)


# hide axis ticks
plt.tick_params(axis="both", which="both", bottom="off", top="off",  
        labelbottom="on", left="off", right="off", labelleft="on")

# remove axis spines
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)    

plt.grid()
plt.tight_layout
plt.show()
#%%

label_dict={1:'A',2:'B',3:'C'}
target_df2=np.array([i[0] for i in target_df['context']])
ax = plt.subplot(111)
for label,marker,color in zip(
    range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

    plt.scatter(x=X_lda[:,0][target_df2 == label_dict[label]],
                y=X_lda[:,1][target_df2 == label_dict[label]], # flip the figure
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label])

plt.xlabel('LD1')
plt.ylabel('LD2')

leg = plt.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.5)


# hide axis ticks
plt.tick_params(axis="both", which="both", bottom="off", top="off",  
        labelbottom="on", left="off", right="off", labelleft="on")

# remove axis spines
ax.spines["top"].set_visible(False)  
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)    

plt.grid()
plt.tight_layout
plt.show()
#%% plot only the first trial
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ctx_id=['A','B','C']
colors=['red','green','blue']

trial_ind=4
for target,color in zip(ctx_id,colors):
    ind2keep=target_df['context']==target+str(trial_ind)
    x=principalDf.loc[ind2keep,'principal component 1']
    y=principalDf.loc[ind2keep,'principal component 2']
    ax.plot(x,y,c=color)         


#%% Rank Sum test of context selectivity 
from scipy.stats import ranksums
T=[]
Pval=[]
for unit in ensemble:   
    fra=unit.ctx[0]['fr']
    frb=unit.ctx[1]['fr']
    t,p=ranksums(fra,frb)
    T.append(t)
    Pval.append(p)
Pval=np.array(Pval)
T=np.array(T)
cell_id=np.where(Pval<=0.05)
num_ctx_cell=len(Pval[Pval<=0.05])
T_ctx=T[Pval<=0.05]
ctxA_unit=[]
ctxB_unit=[]
for ind in cell_id[0]:
    if T[ind]>0:
        ctxA_unit.append(ensemble[ind].animal_id+'_'+ensemble[ind].name)
    elif T[ind]<0:
        ctxB_unit.append(ensemble[ind].animal_id+'_'+ensemble[ind].name)
#%% Pie chart without place cell info
labels='A','B','Others'
sizes=[len(ctxA_unit),len(ctxB_unit),len(ensemble)-len(ctxA_unit)-len(ctxB_unit)]
colors=['red','green','grey']
explode=(0.1,0.1,0.1)
fig=plt.figure()
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Context preference by Ranksum Test, n = %d '%len(ensemble))
plt.show()
#%% bootstrap
ctxselect=[]
for unit in ensemble:
    unit.context_selectivity=bootstrap_context(unit,eval_method='fr1')[0]
    ctxselect.append(unit.context_selectivity)
#%%
labels='A','B','Others'
sizes=[len([cs for cs in ctxselect if cs=='A']),len([cs for cs in ctxselect if cs=='B']),len([cs for cs in ctxselect if cs=='others'])]
colors=['red','green','grey']
explode=(0.1,0.1,0.1)
fig=plt.figure()
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Context preference by Lenient Bootstrap, n = %d '%len(ensemble))
plt.show() 
ctxA_unit_boot=[]
ctxB_unit_boot=[]
for unit in ensemble:
    if unit.context_selectivity=='A':
        ctxA_unit_boot.append(unit.animal_id +'_'+unit.name)
    elif unit.context_selectivity=='B':
        ctxB_unit_boot.append(unit.animal_id +'_'+unit.name)
#%%
placecell=[]
for myUnit in ensemble:
    if myUnit.placefield=='A':
        placecell.append({myUnit.animal_id+'_'+myUnit.name: 'A'})
    elif myUnit.placefield=='B':
        placecell.append({myUnit.animal_id+'_'+myUnit.name: 'B'})
    elif myUnit.placefield=='Both':
        placecell.append({myUnit.animal_id+'_'+myUnit.name: 'Both'})
    
#%% Pie chart
pie_chart_summary_strict_bootstrap(ensemble)

#%%
context_selectivity=[]
for myUnit in ensemble:
    #myUnit.context_selectivity=[]
    context_selectivity.append(bootstrap_context(myUnit,eval_method='cdi',shuffle_num=1000,isfig=False))


#%%
from scipy.stats import variation 

num_ctx_cell=0
cvA=[]
cvB=[]
bins=20
binrange=(0,2)

for myUnit in ensemble:
    if myUnit.context_selectivity=='A' or myUnit.context_selectivity=='B':
        cvA.append(variation(myUnit.ctx[0]['fr']))
        cvB.append(variation(myUnit.ctx[1]['fr']))
        num_ctx_cell+=1
fig,ax=plt.subplots()
plt.hist(cvA,bins,binrange,alpha=0.5,color='r')
plt.hist(cvB,bins,binrange,alpha=0.5,color='b')
plt.xlabel('Coefficient of variation')
plt.ylabel('Count') 
plt.legend(['Context A','Context B'])
plt.title('n=%d'%num_ctx_cell)
#%%
fig,ax=plt.subplots()
plt.scatter(cvA,cvB)
ax.set_xlim([0,1.75])
ax.set_ylim([0,1.75])
x=np.arange(0,1.75,0.1)
ax.plot(x,x,'--')
plt.xlabel('CV in context A')
plt.ylabel('CV in context B')