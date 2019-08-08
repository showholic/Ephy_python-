# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:12:48 2019

@author: Qixin
"""

import nexfile as nex
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy import stats

#%%
pathname=r'C:\Users\Qixin\XuChunLab\nexdata\BaiTao\191545\06072019'
date=os.path.split(pathname)[1]
animal=os.path.split(os.path.split(pathname)[0])[1]
filepath=glob.glob(os.path.join(pathname,'*.nex'))[0]
nexin=nex.Reader(useNumpy=True).ReadNexFile(filepath)
neurons=[]
waveforms=[]
events=[]
markers=[]
for var in nexin['Variables']:
    if var['Header']['Type']==0:
        neurons.append(var)
        #print('neuron',len(neurons))
    if var['Header']['Type']==1:
        events.append(var)
        #print('events',len(events))
    if var['Header']['Type']==3:
        waveforms.append(var)
        #print('waveforms',len(waveforms))
    if var['Header']['Type'] == 6 and len(var['Timestamps']) != 0:
        markers.append(var)
#%%
marker_dict={'Lick left':'EVT09',
             'Lick right':'EVT10',
             'Enter context':'EVT11',
             'Pump right':'EVT05',
             'Pump left':'EVT06',
             'Go cue':'EVT07'
             }
marker_ts={}
for mrker in markers:
    mrker_name=mrker['Header']['Name']
    mrker_ts=mrker['Timestamps']
    for mrker_dict_keys, mrker_dict_value in marker_dict.items():
        if mrker_dict_value==mrker_name:
            marker_ts[mrker_dict_keys]=mrker_ts
marker_ts['Trial start']=events[0]['Timestamps']
marker_ts['Trial end']=events[1]['Timestamps']
#%% set up marker dataframes
def slice_ts(ts_list,t0,t1):
    t_out=ts_list[(ts_list>=t0) & (ts_list<=t1)]
    if ~np.any(t_out):
        t_out=np.nan
    elif np.size(t_out)==1:
        t_out=t_out[0]
    return t_out

def slice_ts2(ts_list,t0,t1):
    t_out=ts_list[(ts_list>=t0) & (ts_list<=t1)]
    if ~np.any(t_out):
        t_out=np.inf
    elif np.size(t_out)==1:
        t_out=t_out[0]
    return t_out

def slice_ts3(ts_list,t0,t1):
    if np.any(ts_list):
        try:
            t_out=ts_list[(ts_list>=t0) & (ts_list<=t1)]
        except:
            t_out=[]

    else:
        t_out=[]
    return t_out

trial_data={'trial start':[],
            'door open':[],
            'enter context':[],
            'enter context2':[],
            'go cue':[],
            'lick time':[],
            'lick choice':[],
            'pump':[],
            'trial end':[],
            'correctness':[],
            'context':[],
            'lick delay':[]
            }
Units=[]
for neu in neurons:
    Units.append({'name':neu['Header']['Name'],'spkt':[]})
    
for t,t_start in enumerate(marker_ts['Trial start']):
    t_end=marker_ts['Trial end'][t]
    trial_data['trial start'].append(t_start)
    trial_data['trial end'].append(t_end)
    trial_data['door open'].append(t_start+1)
    try:
        trial_data['enter context2'].append(slice_ts(marker_ts['Enter context'],t_start,t_end)[0])
    except:
        trial_data['enter context2'].append(np.nan)
    trial_data['go cue'].append(slice_ts(marker_ts['Go cue'],t_start,t_end))
    trial_data['enter context'].append(trial_data['go cue'][t]-1.5)
    lickl=slice_ts2(marker_ts['Lick left'],t_start,t_end)
    lickr=slice_ts2(marker_ts['Lick right'],t_start,t_end)
    lick_compare=[np.min(lickl),np.min(lickr)]
    #we estimated that there was a 0.2s delay of the lick signal
    trial_data['lick time'].append(np.min(lick_compare)+0.2)
    try:
        trial_data['lick delay'].append(trial_data['lick time'][t]-trial_data['go cue'][t])
    except:
        trial_data['lick delay'].append(np.nan)
    trial_data['lick choice'].append(np.argmin(lick_compare))
    pumpl=slice_ts2(marker_ts['Pump left'],t_start,t_end)
    pumpr=slice_ts2(marker_ts['Pump right'],t_start,t_end)
    if ~np.isnan(trial_data['go cue'][t]):      #correctness: 0 is false, 1 is true; context: 0 is A, 1 is B 
        if ~np.isinf(pumpl): #correctA
            trial_data['pump'].append('left')
            trial_data['correctness'].append(1)
            trial_data['context'].append(0)
        elif ~np.isinf(pumpr): #correctB
            trial_data['pump'].append('right')
            trial_data['correctness'].append(1)
            trial_data['context'].append(1)
        else: #no reward        
            if np.argmin(lick_compare)==0 : #incorrectA: licked left but no reward
                trial_data['pump'].append(np.nan)
                trial_data['correctness'].append(0)
                trial_data['context'].append(1)
            elif np.argmin(lick_compare)==1 : #incorrectB: licked right but no reward
                trial_data['pump'].append(np.nan)
                trial_data['correctness'].append(0)
                trial_data['context'].append(0)
            else: #no lick: miss trial
                trial_data['pump'].append(np.nan)
                trial_data['correctness'].append(np.nan)
                trial_data['context'].append(np.nan)
    else: #invalid case
        trial_data['pump'].append(np.nan)
        trial_data['correctness'].append(np.nan)
        trial_data['context'].append(np.nan)
    for n,neu in enumerate(neurons):
        Units[n]['spkt'].append(slice_ts(neu['Timestamps'],t_start,t_end))

trial_df=pd.DataFrame(data=trial_data)
print('done preparing data')
#trial_df=trial_df.dropna(subset=['go cue']) #drop the invalid trials

#%% Raster
def plot_raster(unit,marker_df,event_name,ax,prewindow=0.5,postwindow=0.5,aux_event=[],
                linecolor=[0,0,0],linesize=0.5,shade_invalid=True,drop_invalid=True):
    #raster plot of a single unit
    #aligned by event_name
    #shade the background to indicate auxillary events such as context, lick choice, and correctness
    if drop_invalid:
        valid_index=np.argwhere(~np.isnan(marker_df['go cue'])).ravel()
        spikes=np.asarray(unit['spkt'])[valid_index]
        event=marker_df[event_name][valid_index]
        aux_evt=marker_df[aux_event][valid_index]
    else:
        spikes=unit['spkt']    
        event=marker_df[event_name]
        aux_evt=marker_df[aux_event]
    for t,(spkt,evt) in enumerate(zip(spikes,event)):
        if ~np.isnan(evt): #no event timestamp in this trial -- no idea where to align
            if np.any(spkt): #make sure there is spike in the timewindow
                spktemp=spkt-evt
                spkevent=slice_ts3(spktemp,-prewindow,postwindow)
                if np.any(spkevent):
                    ax.eventplot(spkevent,lineoffsets=t,color=linecolor,linelengths=linesize)
            if aux_event: #e.g: to indicate correct trial and false trial 
                x=[-prewindow,postwindow]
                y1=[t-0.5,t-0.5]
                y2=[t+0.5,t+0.5]
                if aux_evt.iloc[t]==0:
                    ax.fill_between(x, y1, y2 , where=y2>=y1, facecolor='green',alpha=0.5)
                elif aux_evt.iloc[t]==1:
                    ax.fill_between(x, y1, y2 , where=y2>=y1, facecolor='red',alpha=0.5)
        if ~drop_invalid & shade_invalid: #shade the invalid trials
            if np.isnan(marker_df['go cue'][t]):
                    x=[-prewindow,postwindow]
                    y1=[t-0.5,t-0.5]
                    y2=[t+0.5,t+0.5]
                    ax.fill_between(x, y1, y2 , where=y2>=y1, facecolor='grey')
    ax.axvline(x=0,color='g',linestyle='--',linewidth=0.5)
    ax.set_ylim([-1,len(event)])
    ax.set_xlim([-prewindow,postwindow])
    ax.set_title(unit['name'])

def plot_raster_all_unit(Units,trial_df,event2align,prewindow=0.5,postwindow=0.5,aux_event=[],ncol=4,shade_invalid=True,drop_invalid=True):   
    nrow=int(np.ceil(len(Units)/ncol))
    fig,ax=plt.subplots(nrow,ncol,sharex=True)  
    for n,unit in enumerate(Units):
        plot_raster(unit,trial_df,event2align,
                    ax[int(np.floor(n/ncol)),int((n)%ncol)],prewindow,postwindow,aux_event=aux_event,shade_invalid=shade_invalid,drop_invalid=drop_invalid)
    if aux_event:
        plt.suptitle('Raster aligned by ' +event2align + ', colored shades indicate ' + aux_event)
    else:
        plt.suptitle('Raster aligned by ' +event2align)
        


#%%
def plot_aux_PSTH(prewindow,postwindow,tbin,spikes,event,ax,plot_type='line',color='green'):
    bin_edges=np.arange(-prewindow,postwindow+tbin,tbin)
    ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
    fr_binned=[]
    for t,(spkt,evt) in enumerate(zip(spikes,event)):
        if ~np.isnan(evt): #no event timestamp in this trial -- no idea where to align
            if np.any(spkt): #make sure there is spike in the timewindow
                spktemp=spkt-evt
                spkevent=slice_ts3(spktemp,-prewindow,postwindow)
                fr_binned.append(np.divide(np.histogram(spkevent,bin_edges)[0],tbin))
    fr_mean=np.mean(fr_binned,axis=0)
    #fr_error=np.std(fr_binned,axis=0)
    fr_error=stats.sem(fr_binned,axis=0)
    if plot_type=='line':
        ax.plot(ts_bin,fr_mean,color=color)
        ax.fill_between(ts_bin,fr_mean-fr_error,fr_mean+fr_error,alpha=0.4,color=color)
    elif plot_type=='bar':
        ax.bar(ts_bin,fr_mean,width=tbin,edgecolor='k')    

    
def plot_PSTH(unit,marker_df,event_name,ax,prewindow=1,postwindow=1,tbin=0.1,aux_trialind=[],drop_invalid=True,plot_type='line',colors=['red','green']):
    #To make aux_event more versatile, this function requires you to provide the index trial of 2 separate groups 
    if drop_invalid:
        valid_index=np.argwhere(~np.isnan(marker_df['go cue'])).ravel()
        spikes=np.asarray(unit['spkt'])[valid_index]
        event=marker_df[event_name][valid_index]
    else:
        spikes=unit['spkt']    
        event=marker_df[event_name]
        
    if aux_trialind:    
        evt1=marker_df[event_name][aux_trialind[0]]
        evt2=marker_df[event_name][aux_trialind[1]]
        spk1=np.asarray(unit['spkt'])[aux_trialind[0]]
        spk2=np.asarray(unit['spkt'])[aux_trialind[1]]
        plot_aux_PSTH(prewindow,postwindow,tbin,spk1,evt1,ax,plot_type='line',color=colors[0])
        plot_aux_PSTH(prewindow,postwindow,tbin,spk2,evt2,ax,plot_type='line',color=colors[1])
        
        
    else:   
        bin_edges=np.arange(-prewindow,postwindow+tbin,tbin)
        ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
        fr_binned=[]
        for t,(spkt,evt) in enumerate(zip(spikes,event)):
            if ~np.isnan(evt): #no event timestamp in this trial -- no idea where to align
                if np.any(spkt): #make sure there is spike in the timewindow
                    spktemp=spkt-evt
                    spkevent=slice_ts3(spktemp,-prewindow,postwindow)
                    fr_binned.append(np.divide(np.histogram(spkevent,bin_edges)[0],tbin))
        fr_mean=np.mean(fr_binned,axis=0)
        #fr_error=np.std(fr_binned,axis=0)
        fr_error=stats.sem(fr_binned,axis=0)
        if plot_type=='line':
            ax.plot(ts_bin,fr_mean,'k-')
            ax.fill_between(ts_bin,fr_mean-fr_error,fr_mean+fr_error,alpha=0.5)
        elif plot_type=='bar':
            ax.bar(ts_bin,fr_mean,width=tbin,edgecolor='k')    
    ax.axvline(x=0,color='g',linestyle='--',linewidth=0.5)
    ax.set_title(unit['name'])
                
def plot_PSTH_all_unit(Units,trial_df,event2align,prewindow=0.5,postwindow=0.5,tbin=0.1,aux_event='context',aux_id=[],ncol=4,plot_type='line',colors=['red','green']):   
    aux_trialind=get_aux_ind(trial_df,aux_event=aux_event,aux_id=aux_id)

    nrow=int(np.ceil(len(Units)/ncol))
    fig,ax=plt.subplots(nrow,ncol,sharex=True)  
    for n,unit in enumerate(Units):
        plot_PSTH(unit,trial_df,event2align,
                  ax[int(np.floor(n/ncol)),int((n)%ncol)],
                  prewindow=prewindow,postwindow=postwindow,aux_trialind=aux_trialind,tbin=tbin,plot_type=plot_type)
    label_names=get_label_name(aux_event=aux_event,aux_id=aux_id)
    plt.suptitle('PSTH aligned by ' +event2align + ',  '  + label_names[0] + '(' + str(np.size(aux_trialind[0])) + ')' + ' vs ' +label_names[1] + '(' + str(np.size(aux_trialind[1])) + ')')

def get_aux_ind(trial_df,aux_event,aux_id=[]):
    #aux_id should be [] when aux_event has 1 event
    #when aux_event has 2 event:
    #e.g.1: aux_event=['correctness','context'], aux_id=[[1,0],[1,1]] ==> "all correct A trials" and "all correct B trials" 
    #e.g.2: aux_event=['context','correctness'], aux_id=[[1,1],[1,0]] ==> "all context B correct trials" and "all context B incorrect trials" 
    if np.size(aux_event)==2:
        aux_ind=[[],[]]
        aux_ind[0]=np.where((trial_df[aux_event[0]]==aux_id[0][0])&(trial_df[aux_event[1]]==aux_id[0][1]))[0]
        aux_ind[1]=np.where((trial_df[aux_event[0]]==aux_id[1][0])&(trial_df[aux_event[1]]==aux_id[1][1]))[0]
        
    elif np.size(aux_event)==1:   
        aux_ind=[[],[]]
        aux_ind[0]=np.where(trial_df[aux_event]==0)[0]
        aux_ind[1]=np.where(trial_df[aux_event]==1)[0]
    
        
    return aux_ind

def get_label_name(aux_event,aux_id=[]):
    labels=['','']
    if np.size(aux_event)==2:
        labels[0]=event_decoder(aux_event[0],aux_id[0][0]) + ' ' + event_decoder(aux_event[1],aux_id[0][1])
        labels[1]=event_decoder(aux_event[0],aux_id[1][0]) + ' ' + event_decoder(aux_event[1],aux_id[1][1])
    elif np.size(aux_event)==1: 
        labels[0]=event_decoder(aux_event,0)
        labels[1]=event_decoder(aux_event,1)
    return labels
            
                                    
def event_decoder(evt,ind):
    event_dict={'context':{0:'A',1:'B'},'correctness':{0:'False',1:'Correct'},'lick choice':{0:'left',1:'right'}}
    name=event_dict[evt][ind]
    return name


#def set_event_color(aux_event,aux_id=[]):
#    #https://matplotlib.org/gallery/color/colormap_reference.html
#    #the color should be more focused on the second event 
#    aux_color_dict={'context':[plt.cm.RdYlBu(0), plt.cm.RdYlBu(1)],
#                    'correctness':[plt.cm.RdYlGn(0), plt.cm.RdYlGn(1)],
#                    'lick choice':[plt.cm.PiYG(0),plt.cm.PiYG(1)]}
#
#    if np.size(aux_event)==1:    
#        colors=aux_color_dict[aux_event]
#    elif np.size(aux_event)==2:
#        for n,aux_evt in enumerate(aux_event):
#            c1=aux_color_dict[aux_evt][]
#
#    
#    return colors
        


#%% Serial PSTH
def spikevent(spikes,event,prewindow,postwindow,event_offset):
    spkevent=[]
    for t,(spkt,evt) in enumerate(zip(spikes,event)):
        if ~np.isnan(evt): #no event timestamp in this trial -- no idea where to align
            if np.any(spkt): #make sure there is spike in the timewindow
                spktemp=spkt-evt
                try:
                    spkevent.append(slice_ts3(spktemp,-prewindow,postwindow)+event_offset)
                except:
                    spkevent.append([])
    return spkevent
                    
def serial_raster_data_prep(unit,marker_df,event_name,prewindow,postwindow,event_offset=0,aux_trialind=[],drop_invalid=True):        
    if aux_trialind:   
        evt1=marker_df[event_name][aux_trialind[0]]
        evt2=marker_df[event_name][aux_trialind[1]]
        spk1=np.asarray(unit['spkt'])[aux_trialind[0]]
        spk2=np.asarray(unit['spkt'])[aux_trialind[1]]
        if drop_invalid:
            valid_index1=np.argwhere(~np.isnan(evt1)).ravel()
            valid_index2=np.argwhere(~np.isnan(evt2)).ravel()
            spk1=spk1[valid_index1]
            spk2=spk2[valid_index2]
            evt1=evt1.dropna()
            evt2=evt2.dropna()
        spkevent=[spikevent(spk1,evt1,prewindow,postwindow,event_offset),
                      spikevent(spk2,evt2,prewindow,postwindow,event_offset)]
        
    
    elif aux_trialind==[]:
        if drop_invalid:
            valid_index=np.argwhere(~np.isnan(marker_df['go cue'])).ravel()
            spikes=np.asarray(unit['spkt'])[valid_index]
            event=marker_df[event_name][valid_index]
        else:
            spikes=unit['spkt']    
            event=marker_df[event_name]
            
        spkevent=[]
        for t,(spkt,evt) in enumerate(zip(spikes,event)):
            if ~np.isnan(evt): #no event timestamp in this trial -- no idea where to align
                if np.any(spkt): #make sure there is spike in the timewindow
                    spktemp=spkt-evt
                    try:
                        spkevent.append(slice_ts3(spktemp,-prewindow,postwindow)+event_offset)
                    except:
                        spkevent.append([])       
    return spkevent
    
def serial_raster(Units,trial_df,ncol=4,aux_event=[],aux_id=[]):
    nrow=int(np.ceil(len(Units)/ncol))
    fig,axes=plt.subplots(nrow,ncol,sharex=True)  
    fig.set_size_inches(12,8)
    pre1=1
    post1=1.5+ 0.5
    pre2=0.5
    post2=1
    event2_offset=post1+1
    if np.size(aux_event)>=1:
        gap=1
        aux_offset=event2_offset+post2+gap+pre1 #the last digit is the gap 
        aux_trialind=get_aux_ind(trial_df,aux_event,aux_id=aux_id)
        for n,unit in enumerate(Units):
            ax=axes[int(np.floor(n/ncol)),int((n)%ncol)]            
            spkevent1=serial_raster_data_prep(unit,trial_df,'enter context',pre1,post1,event_offset=0,aux_trialind=aux_trialind)
            spkevent2=serial_raster_data_prep(unit,trial_df,'lick time',pre2,post2,event_offset=event2_offset,aux_trialind=aux_trialind)
            for t,(spk1,spk2) in enumerate(zip(spkevent1[0],spkevent2[0])): #plot left
                ax.eventplot(spk1,lineoffsets=t,color='red',linelengths=.5)
                ax.eventplot(spk2,lineoffsets=t,color='red',linelengths=.5)
            for t,(spk1,spk2) in enumerate(zip(spkevent1[1],spkevent2[1])): #plot right
                ax.eventplot(np.add(spk1,aux_offset),lineoffsets=t,color='green',linelengths=.5)
                ax.eventplot(np.add(spk2,aux_offset),lineoffsets=t,color='green',linelengths=.5)
                
            ax.axvline(x=0,color='k',linestyle='--',linewidth=0.5)
            ax.axvline(x=1.5,color='k',linestyle='--',linewidth=0.5)
            ax.axvline(x=event2_offset,color='k',linestyle='--',linewidth=0.5)
            ax.axvspan(post1,event2_offset-pre2,alpha=0.5,color='red')
            ax.axvspan(event2_offset+post2,event2_offset+post2+gap,alpha=0.5,color='gray')
            
            ax.axvline(x=aux_offset,color='k',linestyle='--',linewidth=0.5)
            ax.axvline(x=aux_offset+1.5,color='k',linestyle='--',linewidth=0.5)
            ax.axvline(x=aux_offset+event2_offset,color='k',linestyle='--',linewidth=0.5)
            ax.axvspan(aux_offset+post1,aux_offset+event2_offset-pre2,alpha=0.5,color='green')
            ax.set_xlim([-1,aux_offset+event2_offset+post2])
            plt.setp(ax,xticks=[0,1.5,event2_offset,aux_offset,aux_offset+1.5,aux_offset+event2_offset],xticklabels=['enter','cue','lick','enter','cue','lick'])
            ax.set_title(unit['name'],fontsize=8)
        label_names=get_label_name(aux_event=aux_event,aux_id=aux_id)
        plt.suptitle(label_names[0] + '(' + str(np.size(aux_trialind[0])) + ')' + ' vs ' +label_names[1] + '(' + str(np.size(aux_trialind[1])) + ')',y=1)
        plt.tight_layout()
        
        
        
    else: #no subgroups, plot all trials
        for n,unit in enumerate(Units):
            ax=axes[int(np.floor(n/ncol)),int((n)%ncol)]
            spkevent1=serial_raster_data_prep(unit,trial_df,'enter context',pre1,post1,event_offset=0)
            spkevent2=serial_raster_data_prep(unit,trial_df,'lick time',pre2,post2,event_offset=event2_offset)
            for t,(spk1,spk2) in enumerate(zip(spkevent1,spkevent2)):
                ax.eventplot(spk1,lineoffsets=t,color='k',linelengths=.5)
                ax.eventplot(spk2,lineoffsets=t,color='k',linelengths=.5)
            ax.axvline(x=0,color='g',linestyle='--',linewidth=0.5)
            ax.axvline(x=1.5,color='g',linestyle='--',linewidth=0.5)
            ax.axvline(x=event2_offset,color='g',linestyle='--',linewidth=0.5)
            ax.axvspan(post1,event2_offset-pre2,alpha=0.5,color='gray')
            plt.setp(ax,xticks=[0,1.5,event2_offset],xticklabels=['enter context','go cue','lick'])
            ax.set_title(unit['name'],fontsize=8)
    
    

#%%

    
def plot_aux_PSTH2(tbin,spikes,event,ax,prewindow,postwindow,event_offset=0,plot_type='line',color='green'):
    bin_edges=np.arange(-prewindow,postwindow+tbin,tbin)+event_offset
    ts_bin=np.arange(bin_edges[0]+tbin/2,bin_edges[-1],tbin)
    fr_binned=[]
    for t,(spkt,evt) in enumerate(zip(spikes,event)):
        if ~np.isnan(evt): #no event timestamp in this trial -- no idea where to align
            if np.any(spkt): #make sure there is spike in the timewindow
                spktemp=spkt-evt
                try:
                    spkevent=slice_ts3(spktemp,-prewindow,postwindow)+event_offset
                except: #you cannot do []+event_offset
                    spkevent=[]
                fr_binned.append(np.divide(np.histogram(spkevent,bin_edges)[0],tbin))
    fr_mean=np.mean(fr_binned,axis=0)
    #fr_error=np.std(fr_binned,axis=0)
    fr_error=stats.sem(fr_binned,axis=0)
    if plot_type=='line':
        ax.plot(ts_bin,fr_mean,color=color)
        ax.fill_between(ts_bin,fr_mean-fr_error,fr_mean+fr_error,alpha=0.4,color=color)
    elif plot_type=='bar':
        ax.bar(ts_bin,fr_mean,width=tbin,edgecolor='k')    
    maxfr=np.max(fr_mean)
    return maxfr

def serial_PSTH_single(unit,marker_df,event_name,ax,prewindow=0.5,postwindow=0.5,event_offset=0,tbin=0.1,aux_trialind=[],drop_invalid=True,colors=['red','green']):
    
    if drop_invalid:
            valid_index=np.argwhere(~np.isnan(marker_df['go cue'])).ravel()
            spikes=np.asarray(unit['spkt'])[valid_index]
            event=marker_df[event_name][valid_index]
    else:
        spikes=unit['spkt']    
        event=marker_df[event_name]
    if aux_trialind:    
        evt1=marker_df[event_name][aux_trialind[0]]
        evt2=marker_df[event_name][aux_trialind[1]]
        spk1=np.asarray(unit['spkt'])[aux_trialind[0]]
        spk2=np.asarray(unit['spkt'])[aux_trialind[1]]
    try:
        maxfr1=plot_aux_PSTH2(tbin,spk1,evt1,ax,prewindow=prewindow,postwindow=postwindow,event_offset=event_offset,plot_type='line',color=colors[0])
    except:
        maxfr1=np.nan
        print('no correct trials')
    try:
        maxfr2=plot_aux_PSTH2(tbin,spk2,evt2,ax,prewindow=prewindow,postwindow=postwindow,event_offset=event_offset,plot_type='line',color=colors[1])
    except:
        maxfr2=np.nan
        print('no false trials')
        
    maxfr=np.max([maxfr1,maxfr2])
    return maxfr
    
def serial_PSTH(Units,marker_df,ncol=4,aux_event='context',aux_id=[],colors=['red','green'],new_ylim=[]):
    nrow=int(np.ceil(len(Units)/ncol))
    fig,axes=plt.subplots(nrow,ncol,sharex=True)  
    fig.set_size_inches(12,8)
    pre1=1
    #post1=1.5+np.around(np.mean(trial_df['lick delay']),decimals=1) 
    post1=1.5+0.5
    pre2=0.5
    post2=1
    event2_offset=post1+1
    aux_trialind=get_aux_ind(trial_df,aux_event,aux_id=aux_id)
    maxfr=[]
    ylims=[]
    for n,unit in enumerate(Units):
        ax=axes[int(np.floor(n/ncol)),int((n)%ncol)]
        maxfr1=serial_PSTH_single(unit,marker_df,'enter context',ax,prewindow=pre1,
                           postwindow=post1,event_offset=0,tbin=0.1,
                           aux_trialind=aux_trialind,colors=colors)
        maxfr2=serial_PSTH_single(unit,marker_df,'lick time',ax,prewindow=pre2,
                           postwindow=post2,event_offset=event2_offset,tbin=0.1,
                           aux_trialind=aux_trialind,colors=colors)
        maxfr.append(np.max([maxfr1,maxfr2]))
        ax.axvline(x=0,color='g',linestyle='--',linewidth=0.5)
        ax.axvline(x=1.5,color='g',linestyle='--',linewidth=0.5)
        ax.axvline(x=event2_offset,color='g',linestyle='--',linewidth=0.5)
        ax.axvspan(post1,event2_offset-pre2,alpha=0.3,color='gray')
        plt.setp(ax,xticks=[0,1.5,event2_offset],xticklabels=['enter','cue','lick'])
        ylims.append(ax.get_ylim())
        if new_ylim:
            ax.set_ylim(new_ylim[n])
        ax.set_title(unit['name'],fontsize=8)
        
    label_names=get_label_name(aux_event=aux_event,aux_id=aux_id)
    plt.suptitle(label_names[0] + '(' + str(np.size(aux_trialind[0])) + ')' + ' vs ' +label_names[1] + '(' + str(np.size(aux_trialind[1])) + ')',y=1)
    fig.canvas.set_window_title(label_names[0] + ' vs ' + label_names[1] )
    plt.tight_layout()
    return ylims,maxfr

#%%
if __name__=='__main__':
    
#%% 1. A vs B    
    plt.close('all')
     
    serial_raster(Units,trial_df,aux_event='context')  
    
    ylims1=serial_PSTH(Units,trial_df,ncol=4,aux_event='context') 
#%% 2. A correct vs A false
    serial_raster(Units,trial_df,aux_event=['context','correctness'],aux_id=[[0,1],[0,0]])  
    
    serial_PSTH(Units,trial_df,ncol=4,aux_event=['context','correctness'],aux_id=[[0,1],[0,0]],new_ylim=ylims1[0]) 
#%% 3. B correct vs B false
    serial_raster(Units,trial_df,aux_event=['context','correctness'],aux_id=[[1,1],[1,0]]) 
    
    serial_PSTH(Units,trial_df,ncol=4,aux_event=['context','correctness'],aux_id=[[1,1],[1,0]],new_ylim=ylims1[0])
#%% 4. Correct vs False
    serial_raster(Units,trial_df,aux_event='correctness')
    
    ylims3=serial_PSTH(Units,trial_df,ncol=4,aux_event='correctness')
#%% 5. Correct A vs Correct B
    serial_raster(Units,trial_df,aux_event=['correctness','context'],aux_id=[[1,0],[1,1]])
    
    serial_PSTH(Units,trial_df,ncol=4,aux_event=['correctness','context'],aux_id=[[1,0],[1,1]],new_ylim=ylims3[0])
    
#%% 6. Left vs Right
    serial_raster(Units,trial_df,aux_event='lick choice')
    
    ylims2=serial_PSTH(Units,trial_df,ncol=4,aux_event='lick choice')
    

#%% Save figures to pdf
    save_pdf=False
    if save_pdf:
        import matplotlib.backends.backend_pdf
        import matplotlib._pylab_helpers
        
        pdf_filename=os.path.join(pathname,animal+'_'+date+'_SummaryFigs.pdf')
        pdf=matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
        figures=[manager.canvas.figure 
                 for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        for fig in figures: 
            pdf.savefig(fig)
        pdf.close()
        plt.close('all')


#    plot_raster_all_unit(Units,trial_df,'door open',0.1,0.1,'context')
#    
#    plot_raster_all_unit(Units,trial_df,'enter context',0.5,1.5,'context',drop_invalid=False)
#    #
#    plot_raster_all_unit(Units,trial_df,'go cue',0.5,0.5,'context')
#    
#    plot_raster_all_unit(Units,trial_df,'lick time',0.5,1,'lick choice',drop_invalid=False)
#    
#    #%% PSTH with single auxiliary event
#    
#    #plot_PSTH_all_unit(Units,trial_df,'door open',1,1,0.05,'lick choice')    
#    
#    plot_PSTH_all_unit(Units,trial_df,'enter context',0.5,1.5,0.1,'context')    
#    
#    plot_PSTH_all_unit(Units,trial_df,'go cue',1,1,0.1,'context')
#    
#    plot_PSTH_all_unit(Units,trial_df,'lick time',1,1,0.1,'lick choice')
#    
#    plt.show()
#    
#    #%%
#    plot_PSTH_all_unit(Units,trial_df,'enter context',0.5,1.5,0.1,['correctness','context'],[[1,0],[1,1]])    
#    
#    plot_PSTH_all_unit(Units,trial_df,'enter context',0.5,1.5,0.1,['correctness','context'],[[0,0],[0,1]]) 
   

    #%%
#    #to do: color by the second event 
#     
    
#    serial_PSTH(Units,trial_df,ncol=4,aux_event=['lick choice','correctness'],aux_id=[[0,1],[0,0]],new_ylim=ylims2[0]) 
#    
#    serial_PSTH(Units,trial_df,ncol=4,aux_event=['lick choice','correctness'],aux_id=[[1,1],[1,0]],new_ylim=ylims2[0]) 
#    
#    serial_PSTH(Units,trial_df,ncol=4,aux_event=['correctness','context'],aux_id=[[0,0],[0,1]],new_ylim=ylims3[0])
#    
#    serial_PSTH(Units,trial_df,ncol=4,aux_event=['correctness','context'],aux_id=[[1,0],[1,1]],new_ylim=ylims3[0])