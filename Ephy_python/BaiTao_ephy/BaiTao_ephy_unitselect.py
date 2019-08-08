# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:18:56 2019

@author: Qixin
"""

from BaiTao_ephy2 import *
import matplotlib.patches as mpatches
#%%
pathname=r'C:\Users\Qixin\XuChunLab\nexdata\BaiTao\191545\06072019'
trial_df,Units=data_prep(pathname)


#%%


def single_unit_summary(unit,event,aux_event,aux_id,prewindow=1,postwindow=1.5,tbin=0.1):
    labels=get_label_name(aux_event,aux_id=aux_id)
    fig,ax=plt.subplots(2,1,sharex=True)
    plot_aux_raster(unit,trial_df,event,ax[0],prewindow=prewindow,postwindow=postwindow,aux_event=aux_event,aux_id=aux_id)
    ax[0].set_ylabel('Trial#')
    
    A_patch=mpatches.Patch(color='red',label=labels[0])
    B_patch=mpatches.Patch(color='green',label=labels[1])
    plt.legend(handles=[A_patch,B_patch],loc='upper right',bbox_to_anchor=(1.12, 2.55))
    
    plot_PSTH(unit,trial_df,event,ax[1],prewindow=prewindow,postwindow=postwindow,tbin=tbin,aux_trialind=get_aux_ind(trial_df,aux_event=aux_event,aux_id=aux_id))
    ax[1].set_title('')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Firing rate (Hz)')
    plt.setp(ax[1],xticks=[-prewindow,0,postwindow],xticklabels=[str(-prewindow),event,str(postwindow)])
        
#%%
unit=Units[4]
event='enter context'
prewindow=1
postwindow=1.5
single_unit_summary(unit,event,'context',[],prewindow=prewindow,postwindow=postwindow)
single_unit_summary(unit,event,['correctness','context'],[[0,0],[1,0]],prewindow=prewindow,postwindow=postwindow)
single_unit_summary(unit,event,['correctness','context'],[[0,1],[1,1]],prewindow=prewindow,postwindow=postwindow)
#%%

    