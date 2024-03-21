import torch
import torch.nn as nn

import norse
from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
from norse.torch.module.leaky_integrator import LICell

class SNN_Perceptron_Equalizer(nn.Module):
    def __init__(self, input_features, M, T, hidden_features, output_features,device, dt=0.001):
        super(SNN_Perceptron_Equalizer, self).__init__()
        self.device = device;

        self.input_features  =  input_features
        self.hidden_features =  hidden_features
        self.output_features =  output_features

        self.M               =  M
        self.T               =  T

        self.p1 = LIFParameters(
                    alpha       = torch.full((self.hidden_features,), torch.as_tensor(100.0)).to(device),
                    v_th        = torch.full((self.hidden_features,), torch.as_tensor(  1.0)).to(device),
                    v_leak      = torch.tensor(0.0).to(device),
                    v_reset     = torch.full((self.hidden_features,), torch.as_tensor(  0.0)).to(device),
                    tau_mem_inv = torch.full((self.hidden_features,), torch.as_tensor( 100.0)).to(device),
                    tau_syn_inv = torch.full((self.hidden_features,), torch.as_tensor( 200.0)).to(device))

        self.input_layer    = torch.nn.Linear(self.input_features*self.M , self.hidden_features,bias=False).to(device);
        self.LIFRec_layer   = LIFRecurrentCell(self.hidden_features      , self.hidden_features,p=self.p1,dt=dt,autapses=False).to(device);
        self.output_layer   = torch.nn.Linear(self.hidden_features       , self.output_features,bias=True).to(device);

        self.encoding = nn.Embedding(2**M ,M*T).to(device)
        self.encoding.weight.requires_grad = True

    def __decode_sum(self,x):
        x = torch.sum(x,0);
        return x

    def forward(self, x):
    
        #Encode Quantized Values and reshape to (Time, Batch, EQ_TAP_CNT * Quantizerwordlenght)
        x = self.encoding(x)
        x = x.view( (x.shape[0],self.T,self.M*self.input_features) ).permute(1,0,2)

        s0 = None

        seq_length,batch_size,_ = x.shape

        self.LIFRec_spikes  = torch.zeros(x.shape[0], x.shape[1], self.hidden_features, device=self.device)

        out = torch.zeros(x.shape[0], x.shape[1], self.output_features, device=self.device)

        for ts in range(seq_length):
            z               =  self.input_layer(x[ts,:,:]);
            z,s0            =  self.LIFRec_layer(z,s0);
            self.LIFRec_spikes[ts,:,:]  = z;
            z               =  self.output_layer(z)
            
            out[ts][:][:]   =  z

        hidden_z = self.LIFRec_spikes;
        spikerate = torch.sum(hidden_z)/(hidden_z.shape[0]*hidden_z.shape[1]*hidden_z.shape[2])
        
        z = self.__decode_sum(out)

        del out
        del s0
        del self.LIFRec_spikes
        del hidden_z
        return z, spikerate

