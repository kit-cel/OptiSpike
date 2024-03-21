import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from norse.torch import LIFParameters

import time

import Process_Screen           #Requires Process_Screen.py
import im_dd                    #Requires im_dd.py
import quantizer                #requires quantizer.py

import Equalizer            #Requires Equalizer.py
import quantizer            #requures quantizer.py

#################### Andrej's Lustige Sachen zu .txt Funktion #######################
def save_to_txt(x_in,s="name", label='signal'):
    result = [ l+" " for l in label ]
    result.append("\n")
    x_in = np.transpose(x_in)
    for l in x_in:
        for t in l:
            result.append(str(t)+" ")
        result.append("\n")
    with open('./'+str(s)+'.txt'.format(), 'w') as f:
        f.writelines(result)
######################################################################################


NN_type = "SNN"
EQ_type = "MMSE"

modulation = "PAM"
modulation_order = 2;

channel_type = "im_dd"

Wl_i   = 1550
B_i    = 50*10**9
L_i    = 5000

#ebno_dB = np.arange(12,22,1);                          #EbN0 values at which Symbols will be created
ebno_dB = np.array([14]);                          #EbN0 values at which Symbols will be created

# use pytorch device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for training:",device)

Init_Model_Path = str(2**modulation_order)+"-"+modulation+"_"+str(NN_type)+"_"+str(EQ_type)+"_Equalizer_"+channel_type+".pt";

for idx,snr_i in enumerate(ebno_dB):
    model = torch.load(Init_Model_Path, map_location=torch.device(device));  #initialize model and copy to target device
    model.device = device

    A = torch.linspace(0,2**model.M-1,2**model.M,device=device).int()
    A = model.encoding(A)
    A = A.view( (A.shape[0],model.T,model.M) ).permute(1,0,2)

    classbits = 12
    counts = np.zeros(2**classbits)
    dx,quant = quantizer.midrise(A,classbits,4);
    quant = quant.detach().cpu().numpy()
    for n in range(2**classbits):
        counts[n] = np.count_nonzero(quant==n)
    
    res = np.linspace(-4,4,counts.size)
    res = np.vstack((res,(counts/np.sum(counts)).reshape((1,-1))))

    fig, ax1 = plt.subplots() 
    ax1.set_xlabel('Embedding values') 
    ax1.set_ylabel('Pre training $\%$', color = 'blue') 
    ax1.plot(np.linspace(-4,4,counts.size),counts/np.sum(counts)*100,alpha=0.4,color = 'blue')
    ax1.tick_params(axis ='y', labelcolor = 'blue') 

    path = Init_Model_Path[:-3]+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(snr_i)+"_dB.sd"
    model.load_state_dict(torch.load(path, map_location=torch.device(device)));

    ##w = 1
    ##w = 4
    #w = 8
    ##w = 16
    #dx,quant = quantizer.midrise(model.encoding.weight.data,w,1);
    #model.encoding.weight.data = ((quant-(2**w-1)/2)+0.5)*dx
    #model.encoding.weight.data = model.encoding.weight.detach()/torch.max(torch.abs(model.encoding.weight.detach()),1).values.reshape((-1,1)).repeat(1,model.encoding.weight.detach().shape[1])

    A = torch.linspace(0,2**model.M-1,2**model.M,device=device).int()
    A = model.encoding(A)
    A = A.view( (A.shape[0],model.T,model.M) ).permute(1,0,2)

    plt.figure(1)
    classbits = 12
    counts = np.zeros(2**classbits)
    dx,quant = quantizer.midrise(A,classbits,4);
    quant = quant.detach().cpu().numpy()
    for n in range(2**classbits):
        counts[n] = np.count_nonzero(quant==n)
    
    res = np.vstack((res,(counts/np.sum(counts)).reshape((1,-1))))


    ax2 = ax1.twinx() 
    ax2.set_ylabel('Trained $\%$', color = 'red') 
    ax2.plot(np.linspace(-4,4,counts.size),counts/np.sum(counts)*100,alpha=1.0,color='red')
    ax2.tick_params(axis ='y', labelcolor = 'red') 

    plt.show()

