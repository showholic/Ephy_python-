# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:15:09 2019

@author: Qixin
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from build_unit import *
#%%
pathname=r'C:\Users\Qixin\OneDrive\Lab_stuff\EPHY_Qiushou\Ctx_discrimination_data\All'
filelist=glob.glob(os.path.join(pathname,'*.pkl'))
ensemble=[]
for filename in filelist:
    ensemble.append(pd.read_pickle(filename)) 
#%% Summary 
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
for filename in filelist:
    unit=pd.read_pickle(filename)
    if unit.is_place_cell:
        num_place_cell+=1
        if unit.context_selectivity[0]!='others':
            num_ctx_and_place_cell+=1
    if unit.context_selectivity[0]!= 'others':
        #print(unit.context_selectivity[0])
        num_ctx_cell+=1
        if (unit.context_selectivity[0].upper())=='A':
            num_A+=1
            if unit.is_place_cell:
                num_AP+=1
        if (unit.context_selectivity[0].upper())=='B':
            num_B+=1
            if unit.is_place_cell:
                num_BP+=1
    else:
        num_non_ctx_cell+=1
        if unit.is_place_cell:
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
plt.show()
#%%
real_binsize=1
sigma=3
speed_threshold=2
for filename in filelist:
    unit=pd.read_pickle(filename)
    if unit.is_place_cell:
        #plot_place_field(unit,real_binsize=1,sigma=2,speed_threshold=5)
        #print(bootstrap_context(unit,eval_method='cdi',shuffle_num=5000,isfig=True))
        plot_ctxwise_PF(unit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
        
#%% Use a different bootstrap method 
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
for filename in filelist:
    unit=pd.read_pickle(filename)
    CS=bootstrap_context(unit,eval_method='cdi',shuffle_num=5000,isfig=False)
    if unit.is_place_cell:
        num_place_cell+=1
        if CS!='others':
            num_ctx_and_place_cell+=1
    if CS!= 'others':
        #print(unit.context_selectivity[0])
        num_ctx_cell+=1
        if (CS.upper())=='A':
            num_A+=1
            if unit.is_place_cell:
                num_AP+=1
        if (CS.upper())=='B':
            num_B+=1
            if unit.is_place_cell:
                num_BP+=1
    else:
        num_non_ctx_cell+=1
        if unit.is_place_cell:
            num_OP+=1
#%%
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