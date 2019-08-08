# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:28:35 2019

@author: Qixin
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from build_unit import *
from scipy.stats import zscore
from scipy.signal import find_peaks
from scipy import stats
pathname=r'C:\Users\Qixin\OneDrive\Lab_stuff\EPHY_Qiushou\Ctx_discrimination_data\All'
filelist=glob.glob(os.path.join(pathname,'*.pkl'))

#Combine units into a list called ensemble
ensemble=[]
for filename in filelist:
    ensemble.append(pd.read_pickle(filename)) 

#%%
####parameters initialization####
entrydur_range=[0,50]
tbin=0.1 #firing rate average over tbin second window 
preevent=5
postevent=5
spkthreshold=20 #minimum spike count required
sigthreshold=1.65 #threshold for zscore response

event_response={'A':{'sig':[]},
                'B':{'sig':[]}}
for n,myUnit in enumerate(ensemble):
    #clean up unit marker, overwrite manual labeled event marker with auto detected ones

    trial=myUnit.split_trial()
    door_marker={'open':[ti['door_open'] for ti in trial],
                 'close':[ti['door_close'] for ti in trial],
                 'duration':myUnit.marker.door['duration']}
    event=door_marker['open'] #aligned by door open 
    response_temp={'A':{'sig':[],'delay':[]},
                        'B':{'sig':[],'delay':[]}}
    for num_trial,ctx in enumerate(myUnit.marker.protocol):
    
        if (entrydur_range[0]<=door_marker['duration'][num_trial]<=entrydur_range[1]): #first constraint on entry duration, second constraint on only using 1 trial for each context
            spkt=trial[num_trial]['spkt'] 
            spkt=np.extract(spkt<door_marker['close'][num_trial]+10,spkt) #only need spikes until 10s after door close                        
            if len(spkt)>=spkthreshold:
                ifr,ts=binned_FR(spkt,door_marker['close'][num_trial]+10,tbin,sigma=5) #turn spike train into continuous firing rate trace by averaging spike counts over tbin seconds, and smooth the trace with 1d Gaussian filter with sigma=5  
                zfr=zscore(ifr) #zscore normalization 
                z_event=np.extract(ts>=event[num_trial]-preevent,zfr)[:int((preevent+postevent)/tbin)+1] #extract peri-event signal 
                ts_event=np.extract(ts>=event[num_trial]-preevent,ts)[:int((preevent+postevent)/tbin)+1]-event[num_trial] 
                sig_t=np.extract(z_event>=sigthreshold,ts_event) #detect time points with significant activation 
                if (sig_t.size>0) & (np.max(z_event)<=20) & (ctx=='A'):
                    response_temp[ctx]['sig'].append(z_event)
                    response_temp[ctx]['delay'].append(sig_t[0])
                elif ctx=='B':
                    response_temp[ctx]['sig'].append(z_event)
                    try:
                        response_temp[ctx]['delay'].append(sig_t[0])
                    except:
                        response_temp[ctx]['delay'].append([])
                else:
                    pass
            
    if (np.size(response_temp['A']['delay'])>0) &(np.size(response_temp['B']['sig'])>0):
        event_response['A']['sig'].append(response_temp['A']['sig'][0]) 
        event_response['B']['sig'].append(response_temp['B']['sig'][0]) 
    else:
        pass

f,ax=plt.subplots(2,1,sharex=True)
for sig in event_response['A']['sig']:    
    ax[0].plot(ts_event,sig,color='r')
for sig in event_response['B']['sig']:    
    ax[0].plot(ts_event,sig,color='b')
plt.title('n= %d'%len(event_response['A']['sig']))
colors=['red','blue']
for ctx,color in zip(['A','B'],colors):
    sigmean=np.mean(event_response[ctx]['sig'],axis=0)
    sigerror=stats.sem(event_response[ctx]['sig'],axis=0)
    ax[1].plot(ts_event,sigmean,color=color)
    ax[1].fill_between(ts_event,sigmean+sigerror,sigmean-sigerror,color=color,alpha=0.5)
ax[1].set_xticks([-preevent,0,postevent])
ax[1].set_xticklabels([-preevent,'door open',postevent])
ax[1].set_ylabel('z-score')
plt.legend(['ContextA','ContextB'])

#%%
