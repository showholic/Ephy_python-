# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:22:30 2019

@author: Qixin
"""

import numpy as np
import matplotlib.pyplot as plt
from build_unit import *
from scipy.ndimage.filters import gaussian_filter1d
from scipy import interpolate

#%%
def speed_corr(Unit,num_context,num_trial,ax,tbin=0.25,isfig=True):
#this function plot correlation with speed for a single trial in a given context
    c=num_context
    t=num_trial
    spk=Unit.ctx[c]['spkt_raw'][t]
    pos=Unit.ctx[c]['pos'][t]
    s=pos['speed'].values
#    ctx_start=Unit.ctx[c]['ctx_start'][t]
#    ctx_end=Unit.ctx[c]['ctx_end'][t]
    ts=pos['ts'].values
    
    #bin spikes and filter firing rate
    bins=np.arange(ts[0],ts[-1],tbin)
    spkc,bin_edges=np.histogram(spk,bins)
    ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
    fr=spkc/tbin
    frf=gaussian_filter1d(fr,tbin/(1/Unit.pos['FPS']))
    
    # filter speed 
    fs=interpolate.interp1d(ts,s,fill_value="extrapolate",kind='linear')
    sfinterp=gaussian_filter1d(fs(ts_bin),tbin/(1/Unit.pos['FPS']))
    
    corr_coef=np.corrcoef(frf,sfinterp)[0][1]
    if isfig:
        ax2=ax.twinx()
        plt.axis('normal')
        ax.plot(ts_bin,frf,color='orange')
        ax.set_ylabel('Firing rate (Hz)')
        ax2.plot(ts_bin,sfinterp-5,color='blue',linewidth=0.5,alpha=0.5)
        ax2.set_ylim([-10,40])
        ax2.set_yticks([])
        plt.show()
        plt.title('corr = %.2f' % corr_coef)
    
    return corr_coef

#%%
def speed_corr_summary(Unit,tbin=0.25,isfig=True):
    corr_coef=[[],[]]
    if isfig:
        f, ax = plt.subplots(2,4, sharey=True)
        f.set_size_inches(8,2)
    for c in range(2):
        for t in range(len(Unit.ctx[c]['ctx_start'])):
            corr_coef[c].append(speed_corr(Unit,c,t,ax[c,t],tbin=tbin,isfig=isfig))
    return corr_coef

#%%
if __name__=='__main__':
    flag_buildtrack=True
    #ensemble,pos=buildneurons(build_tracking=flag_buildtrack,arena_reset=False)
    pathname=r'D:\EPHY_Qiushou\Ctx_discrimination_data\192089'  
    ftype='nex'
    ensemble,pos=buildneurons(pathname,
                              file_type=ftype,
                              build_tracking=True,
                              arena_reset=True,
                              body_part='Body',
                              bootstrap=False,
                              filter_speed=False                              
                              )
    #%%
    corr_coef=speed_corr_summary(ensemble[0])