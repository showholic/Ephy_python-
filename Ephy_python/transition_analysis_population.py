# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:55:33 2019

@author: Qixin
"""
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from build_unit import *
from scipy.stats import zscore
from draw_arena import set_arena
from scipy.signal import find_peaks
from scipy import stats

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

#%% fast trials
dur_range=[0,200]
postdoor=5
tbin=0.1
preevent=60
postevent=5
spkthreshold=50
sigthreshold=1.65
event_response={'A':{'sig':[],'delay':[],'trial':[]},
                'B':{'sig':[],'delay':[],'trial':[]}}
for n,myUnit in enumerate(ensemble):
    ia=1
    ib=1
    try:
        trial=myUnit.split_trial()
    except:
        doormarker=myUnit.marker.door
        myUnit.marker.door={'open':[],'close':[],'duration':[]}
        myUnit.marker.door['open']=doormarker[0]
        myUnit.marker.door['close']=doormarker[1]
        myUnit.marker.door['duration']=doormarker[1]-doormarker[0]
        trial=myUnit.split_trial()
    try:
        dur= myUnit.marker.door['duration']
    except:
        myUnit.marker.door['duration']=myUnit.marker.door['close']-myUnit.marker.door['open']
    door_marker=myUnit.marker.door
    event=door_marker['open']
    for num_trial in range(len(trial)):
        if (dur_range[0]<=door_marker['duration'][num_trial]<=dur_range[1]) & (ia==1 or ib==1):
            spkt=trial[num_trial]['spkt']
            spkt=np.extract(spkt<door_marker['close'][num_trial]+postdoor,spkt)
            if len(spkt)>=spkthreshold:                            
                ifr,ts=binned_FR(spkt,door_marker['close'][num_trial]+postdoor,tbin,sigma=5)
                ifr_baseline=np.extract((ts>=event[num_trial]-preevent)&(ts<event[num_trial]),ifr)
                z_baseline=zscore(ifr_baseline)
                mean_baseline=np.nanmean(ifr_baseline)
                ts_baseline=np.extract((ts>=event[num_trial]-preevent)&(ts<event[num_trial]),ts)-event[num_trial]
                std_baseline=np.nanstd(ifr_baseline)
                ifr_event=np.extract((ts>=event[num_trial]),ifr)[:int((postevent)/tbin)+1]
                z_event=np.divide((ifr_event-mean_baseline),std_baseline)
                ts_event=np.extract((ts>=event[num_trial]),ts)[:int((postevent)/tbin)+1]-event[num_trial]
                sig_t=np.extract(z_event>=sigthreshold,ts_event)
                if (sig_t.size>0) & (np.max(z_event)<=20):
                    if trial[num_trial]['name']=='A':
                        event_response['A']['sig'].append(z_event)
                        event_response['A']['delay'].append(sig_t[0])
                        event_response['A']['trial'].append(myUnit.animal_id+myUnit.name)
                        ia=0
                    elif trial[num_trial]['name']=='B':
                        event_response['B']['sig'].append(z_event)
                        event_response['B']['delay'].append(sig_t[0])
                        event_response['B']['trial'].append(myUnit.animal_id+myUnit.name)
                        ib=0
            else:
                pass 
f,ax=plt.subplots(2,1,sharex=True)
for sig in event_response['A']['sig']:    
    ax[0].plot(ts_event,sig,color='r')
for sig in event_response['B']['sig']:    
    ax[0].plot(ts_event,sig,color='b')

colors=['red','blue']
for ctx,color in zip(['A','B'],colors):
    sigmean=np.mean(event_response[ctx]['sig'],axis=0)
    sigerror=stats.sem(event_response[ctx]['sig'],axis=0)
    ax[1].plot(ts_event,sigmean,color=color)
    ax[1].fill_between(ts_event,sigmean+sigerror,sigmean-sigerror,color=color,alpha=0.5)
#%%            
both=set(event_response['A']['trial']).intersection(event_response['B']['trial'])
sigboth={'A':[],
         'B':[]}
sigboth['A']=[event_response['A']['sig'][i] for i in [event_response['A']['trial'].index(x) for x in both]]
sigboth['B']=[event_response['B']['sig'][i] for i in [event_response['B']['trial'].index(x) for x in both]]
f,ax=plt.subplots(2,1,sharex=True)
for sig in sigboth['A']:    
    ax[0].plot(ts_event,sig,color='r')
for sig in sigboth['B']:    
    ax[0].plot(ts_event,sig,color='b')
plt.title('n= %d'%len(both))
colors=['red','blue']
for ctx,color in zip(['A','B'],colors):
    sigmean=np.mean(sigboth[ctx],axis=0)
    sigerror=stats.sem(sigboth[ctx],axis=0)
    ax[1].plot(ts_event,sigmean,color=color)
    ax[1].fill_between(ts_event,sigmean+sigerror,sigmean-sigerror,color=color,alpha=0.5)   
ax[1].set_xticks([0,postevent])
ax[1].set_xticklabels(['door open',postevent])
ax[1].set_ylabel('z-score')
plt.legend(['ContextA','ContextB'])
#%% slow trials 
dur_range=[0,40]
predoor=60
postdoor=10
tbin=0.1
preevent=5
postevent=5
spkthreshold=50
sigthreshold=1.65
event_response={'A':{'sig':[],'delay':[],'trial':[]},
                'B':{'sig':[],'delay':[],'trial':[]}}
for n,myUnit in enumerate(ensemble):
    ia=1
    ib=1
    try:
        trial=myUnit.split_trial()
    except:
        doormarker=myUnit.marker.door
        myUnit.marker.door={'open':[],'close':[],'duration':[]}
        myUnit.marker.door['open']=doormarker[0]
        myUnit.marker.door['close']=doormarker[1]
        myUnit.marker.door['duration']=doormarker[1]-doormarker[0]
        trial=myUnit.split_trial()
    try:
        dur= myUnit.marker.door['duration']
    except:
        myUnit.marker.door['duration']=myUnit.marker.door['close']-myUnit.marker.door['open']
    door_marker=myUnit.marker.door
    event=door_marker['close']
    for num_trial in range(len(trial)):
        if (dur_range[0]<=door_marker['duration'][num_trial]<=dur_range[1]) & (ia==1 or ib==1):
            spkt=trial[num_trial]['spkt']
            spkt=np.extract(spkt<door_marker['close'][num_trial]+postdoor,spkt)
            if len(spkt)>=spkthreshold:                            
                ifr,ts=binned_FR(spkt,door_marker['close'][num_trial]+postdoor,tbin,sigma=5)
                ifr_baseline=np.extract((ts>=door_marker['open'][num_trial]-predoor)&(ts<door_marker['open'][num_trial]),ifr)
                z_baseline=zscore(ifr_baseline)
                mean_baseline=np.nanmean(ifr_baseline)
                std_baseline=np.nanstd(ifr_baseline)
                
                ifr_event=np.extract((ts>=event[num_trial]-preevent),ifr)[:int((preevent+postevent)/tbin)+1]
                z_event=np.divide((ifr_event-mean_baseline),std_baseline)
                ts_event=np.extract((ts>=event[num_trial]-preevent),ts)[:int((preevent+postevent)/tbin)+1]-event[num_trial]
                sig_t=np.extract(z_event>=sigthreshold,ts_event)
                if (sig_t.size>0) & (np.max(z_event)<=20):
                    if trial[num_trial]['name']=='A':
                        event_response['A']['sig'].append(z_event)
                        event_response['A']['delay'].append(sig_t[0])
                        event_response['A']['trial'].append(myUnit.animal_id+myUnit.name)
                        ia=0
                    elif trial[num_trial]['name']=='B':
                        event_response['B']['sig'].append(z_event)
                        event_response['B']['delay'].append(sig_t[0])
                        event_response['B']['trial'].append(myUnit.animal_id+myUnit.name)
                        ib=0
            else:
                pass 

        
f,ax=plt.subplots(2,1,sharex=True)
for sig in event_response['A']['sig']:    
    ax[0].plot(ts_event,sig,color='r')
for sig in event_response['B']['sig']:    
    ax[0].plot(ts_event,sig,color='b')

colors=['red','blue']
for ctx,color in zip(['A','B'],colors):
    sigmean=np.mean(event_response[ctx]['sig'],axis=0)
    sigerror=stats.sem(event_response[ctx]['sig'],axis=0)
    ax[1].plot(ts_event,sigmean,color=color)
    ax[1].fill_between(ts_event,sigmean+sigerror,sigmean-sigerror,color=color,alpha=0.5)
#%% 
both=set(event_response['A']['trial']).intersection(event_response['B']['trial'])
sigboth={'A':[],
         'B':[]}
sigboth['A']=[event_response['A']['sig'][i] for i in [event_response['A']['trial'].index(x) for x in both]]
sigboth['B']=[event_response['B']['sig'][i] for i in [event_response['B']['trial'].index(x) for x in both]]
f,ax=plt.subplots(2,1,sharex=True)
for sig in sigboth['A']:    
    ax[0].plot(ts_event,sig,color='r')
for sig in sigboth['B']:    
    ax[0].plot(ts_event,sig,color='b')
ax[0].set_ylabel('z-score')
plt.title('n= %d'%len(both))    
colors=['red','blue']
for ctx,color in zip(['A','B'],colors):
    sigmean=np.mean(sigboth[ctx],axis=0)
    sigerror=stats.sem(sigboth[ctx],axis=0)
    ax[1].plot(ts_event,sigmean,color=color)
    ax[1].fill_between(ts_event,sigmean+sigerror,sigmean-sigerror,color=color,alpha=0.5)
ax[1].set_xticks([-preevent,0,postevent])
ax[1].set_xticklabels([-preevent,'door close',postevent])
ax[1].set_ylabel('z-score')
plt.legend(['ContextA','ContextB'])
