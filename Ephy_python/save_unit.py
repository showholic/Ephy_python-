# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 19:21:26 2019

@author: Qixin
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from build_unit import *
from scipy.stats import zscore
import pickle
#%%
flag_buildtrack=True
    #ensemble,pos=buildneurons(build_tracking=flag_buildtrack,arena_reset=False)
pathname=r'C:\Users\Qixin\OneDrive\Lab_stuff\EPHY_Qiushou\Ctx_discrimination_data\192087\20190609'  
ftype='nex'
ensemble,pos=buildneurons(pathname,
                          file_type=ftype,
                          build_tracking=flag_buildtrack,
                          arena_reset=True,
                          body_part='Body',
                          bootstrap=False                              
                          )
#%%
real_binsize=1
sigma=2
speed_threshold=3

import matplotlib.backends.backend_pdf
import matplotlib._pylab_helpers
plt.close('all') #make sure everything's closed
for i,myUnit in enumerate(ensemble): 
    
    myUnit.waveform.Plotwf() 
    spkhist_wholetrial(myUnit,tbin=1,sigma=1)
    plot_context_summary(myUnit)      
    #Plot the spike map 
    plot_spikemap(myUnit,speed_threshold=speed_threshold)       
    try:
        myUnit.FR_map,myUnit.SI,myUnit.pSI,myUnit.placefield=plot_place_field(myUnit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)  
    except:
        print('no place field')
    try:
        plot_ctxwise_PF(myUnit,real_binsize=real_binsize,sigma=sigma,speed_threshold=speed_threshold)
    except:
        print('no place field')
    figures=[manager.canvas.figure 
                 for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    pdf_filename=os.path.join(pathname,myUnit.animal_id+'_'+myUnit.date+'_'+myUnit.name  +'_SummaryFigs.pdf')
    pdf=matplotlib.backends.backend_pdf.PdfPages(pdf_filename)
    for fig in figures: 
        pdf.savefig(fig)
    pdf.close()
    plt.close('all')
    if myUnit.max_spkc_ctx>=50:
        filename=os.path.join(pathname, myUnit.date +'_'+ myUnit.animal_id +'_'+ str(i)+ '.pkl')
        with open(filename,'wb') as output:
            pickle.dump(myUnit,output,pickle.HIGHEST_PROTOCOL)
        