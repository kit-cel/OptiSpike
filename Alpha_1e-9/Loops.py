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

####################################################################################################################
#                                           Helper                                                                 #
####################################################################################################################

# L_p L_q regularization to ensure sparse Embedding
def lplq_reg(x):
    denum = torch.sum(torch.abs(x)**2+1e-20,axis=1)
    p_n   = torch.abs(x+1e-20)**2/denum.reshape((-1,1)).repeat(1,x.shape[1])
    lplq  = torch.sum( torch.sqrt(p_n) ,axis=1)
    return lplq

####################################################################################################################
#                                           Training                                                               #
####################################################################################################################

def SNN_MMSE_Symbols_to_Batch(recv_symbols       ,EQ_TAP_CNT,one_hot_encoder,device):
    rollsym = torch.hstack( (recv_symbols[-EQ_TAP_CNT//2+1::],recv_symbols) )
    rollsym = torch.hstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:]) )
    in_sym  = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()

    #Quantization for learned Embedding
    d_x,quant_input = quantizer.midrise(in_sym, 8, 4)
    inputs = quant_input.int()

    return inputs

def Train(Init_Model_Path, 
          NN_type, EQ_type,
          BAUDRATE, L, D, wavelen, SPS, beta, BIAS,
          ebno_dB, modem, EQ_TAP_CNT,
          EPOCHS, BATCHES_PER_SNR, BATCHSIZE_MIN, BATCHSIZE_MAX, LR, GAMMA):

    t_st = time.time()

    modulation_order = modem.give_mod_order();
    modulation       = modem.give_modulation();
    gray             = modem.is_gray_encoding();
    IMDD             = modem.give_IMDD();
    LMQ              = modem.give_LMQ();
    device           = modem.give_device();

    softmax = nn.Softmax(dim=1)
 
    one_hot_encoder = None
 
    monitor = Process_Screen.Screen(PATH = Init_Model_Path, 
                     BAUDRATE=BAUDRATE, L=L, D=D, wavelen=wavelen,
                     BAUDRATE_TOTAL=BAUDRATE, L_TOTAL=L, wavelen_TOTAL=wavelen,SNR_TOTAL=ebno_dB,
                     SPS=SPS,beta=beta, BIAS=BIAS,
                     modulation_order=modulation_order,modulation=modulation,gray=gray,IMDD=IMDD,LMQ=LMQ,
                     LR=LR,gamma=GAMMA,
                     NN_type=NN_type,EQ_type=EQ_type)

    loss_fn = torch.nn.CrossEntropyLoss()

    Symbol_energy = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
    Noise_energy  = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
    SER           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
    BER           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));

    monitor.EPOCHS=EPOCHS
    monitor.TOTAL_MAX=EPOCHS*np.sum(BATCHES_PER_SNR)*wavelen.size*BAUDRATE.size*L.size*ebno_dB.size

    for idx_Wl, Wl_i in enumerate(wavelen):
        monitor.wavelen = Wl_i;
        for idx_B, B_i in enumerate(BAUDRATE):
            monitor.B = B_i
            for idx_L, L_i in enumerate(L):
                monitor.L = L_i
                for idx,snr_i in enumerate(ebno_dB):
                    train_loss = [];

                    BATCHES = BATCHES_PER_SNR[idx]
                    monitor.BATCHES = BATCHES
                    monitor.SNR = snr_i

                    model = torch.load(Init_Model_Path, map_location=torch.device(device));  #initialize model and copy to target device

                    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

                    batch_size_per_epoch = np.linspace(BATCHSIZE_MIN,BATCHSIZE_MAX,num=EPOCHS).astype(int)  #Create list of batch sizes
                    batch_size_per_epoch = batch_size_per_epoch//2 * 2                   #Ensure all batch sizes are even

                    for epoch_cnt in range(EPOCHS):
                        e_st = time.time()
                        monitor.EPOCH = epoch_cnt+1;

                        sym_cnt = int(batch_size_per_epoch[epoch_cnt])                  #Number of symbols
                        N       = modulation_order*sym_cnt                              #Number of bits
                        monitor.Batchsize = sym_cnt

                        channel = im_dd.IM_DD_Channel(sym_cnt,B_i,modulation_order,SPS,D,L_i,Wl_i,beta,snr_i,BIAS,device);

                        for batch_cnt in range(BATCHES):
                            monitor.BATCH = batch_cnt+1;
                            monitor.TOTAL_IDX=(epoch_cnt+1)*(batch_cnt+1)*(idx_Wl+1)*(idx_B+1)*(idx_L+1)*(idx+1)

                            b_st = time.time()

                            ##################################### Bits ################################################

                            bits = torch.randint(0,2,size=(N,1),device=device).flatten();

                            ##################################### Sender ################################################

                            A,symbols = modem.modulate(bits);
                            symbols = symbols * np.sqrt(5)
                            rate = modulation_order;
                            esn0_dB = snr_i + 10 * np.log10( rate )

                            #################################### Channel  ################################################

                            recv_symbols = channel.apply(symbols);
                            recv_symbols = recv_symbols - torch.mean(recv_symbols);
                            e_sym,e_noise = channel.give_energies();

                            Symbol_energy[idx_Wl][idx_B][idx_L][idx] += e_sym;
                            Noise_energy[idx_Wl][idx_B][idx_L][idx] += e_noise;

                            #################################### Create batch  ###########################################
                        
                            labels = A.to(torch.int64)

                            inputs = SNN_MMSE_Symbols_to_Batch(recv_symbols        ,EQ_TAP_CNT,one_hot_encoder,device);

                            #################################### Check intermediate performance  #########################

                            if(batch_cnt % np.ceil(BATCHES/10) == 0):
                                outputs, reg = model(inputs)
                                alpha = 1e-9
                                loss  = (1-alpha)*loss_fn(outputs, labels)                                            #CE Loss
                                loss += alpha*torch.mean(lplq_reg(model.encoding.weight))            #LpLq Regularization (Sparsity)
                                A_eq = torch.argmax(softmax(outputs),axis=1).int();

                                d_x, recv_bits = modem.A_to_bits(A_eq);

                                classerr = torch.sum(torch.where(A_eq == A,0,1))
                                monitor.SER = classerr/A.shape[0]*100

                                test = torch.sum(torch.abs(recv_bits-bits))
                                monitor.BER = test/bits.shape[0]*100
                                monitor.LOSS = loss
                                monitor.REG = reg
                                monitor.show()
                                print(torch.mean(torch.abs(lplq_reg(model.encoding.weight)-1)).detach().cpu())            #LpLq Regularization (Sparsity)
 
                                del recv_bits
                                del A_eq
                                    
                            #################################### Update weights  ###########################################

                            outputs,reg = model(inputs)
                            alpha = 1e-9
                            loss  = (1-alpha)*loss_fn(outputs, labels)                                            #CE Loss
                            loss += alpha*torch.mean(lplq_reg(model.encoding.weight))            #LpLq Regularization (Sparsity)

                            train_loss.append( loss.detach().cpu().numpy() );

                            # compute gradient
                            loss.backward()
                                            
                            # optimize
                            optimizer.step()
                                                        
                            # reset gradients
                            optimizer.zero_grad()

                            b_et = time.time()
                            monitor.Time_per_Batch = b_et-b_st;

                            #Normalize Encoding Weights
                            model.encoding.weight.data = model.encoding.weight.detach()/torch.max(torch.abs(model.encoding.weight.detach()),1).values.reshape((-1,1)).repeat(1,model.encoding.weight.detach().shape[1])

                            del inputs
                            del labels
                            del loss
                            del outputs
                            del reg
                            del symbols
                            del recv_symbols
                            del e_sym
                            del e_noise

                        e_et = time.time()
                        monitor.Time_per_Epoch = e_et-e_st;

                        SER[idx_Wl][idx_B][idx_L][idx] = classerr/A.shape[0]
                        BER[idx_Wl][idx_B][idx_L][idx] = test/bits.shape[0]
                       
                        monitor.LR = LR
                        del channel
                        del classerr
                        del A
                        del test
                        del bits

                    torch.save(model.state_dict(),Init_Model_Path[:-3]+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(snr_i)+"_dB.sd");

                    BER[idx_Wl][idx_B][idx_L][idx] = BER[idx_Wl][idx_B][idx_L][idx]/EPOCHS
                    SER[idx_Wl][idx_B][idx_L][idx] = SER[idx_Wl][idx_B][idx_L][idx]/EPOCHS
                    del model

                    plt.figure(1)
                    plt.semilogy(train_loss);
                    plt.title("Training Loss");
                    plt.xlabel("Optimization Steps over all Epochs")
                    plt.ylabel("Loss")
                    mng = plt.get_current_fig_manager()
                    mng.full_screen_toggle()
                    plt.savefig("Training_Loss_"+str(NN_type)+"_"+str(EQ_type)+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(snr_i)+"_dB.svg", format="svg")
                    plt.clf()

    t_et = time.time()
    print()
    print("Time to train: "+str(t_et-t_st)+"\n")

    #Write results to file
    fp = open("Training"+str(NN_type)+"_"+str(EQ_type)+".txt", 'w')
    fp.write("EbN0: "+str(ebno_dB)+"\n")
    fp.write("BER: "+str(BER)+"\n")
    fp.write("SER: "+str(SER)+"\n")
    fp.write("Noisepower: "+str(-10*np.log10(Noise_energy/(SPS*np.sum(batch_size_per_epoch)*BATCHES)))+"\n")
    fp.write("SNR: "+str(10*np.log10(Symbol_energy/(Noise_energy)))+"\n")
    fp.write("Time to train: "+str(t_et-t_st)+"\n")
    fp.close()

####################################################################################################################
#                                           Evaluation                                                             #
####################################################################################################################
def Operate_SNN_MMSE(model,recv_symbols, EQ_TAP_CNT, monitor, device):
    softmax = nn.Softmax(dim=1)

    sym_cnt = recv_symbols.shape[0]
    EPOCHS  = recv_symbols.shape[1]

    A_eq = torch.zeros( (sym_cnt,EPOCHS), device=device);
    REG  = torch.zeros( (sym_cnt,EPOCHS), device=device);
    
    monitor.EPOCHS=EPOCHS
    
    for epoch_cnt in range(EPOCHS):
        rollsym = torch.hstack( (recv_symbols[-EQ_TAP_CNT//2+1::,epoch_cnt],recv_symbols[::,epoch_cnt]) )
        rollsym = torch.hstack( (rollsym,recv_symbols[:EQ_TAP_CNT//2:,epoch_cnt]) )
        in_sym  = rollsym.unfold(dimension=0, size=EQ_TAP_CNT, step=1 ).float()
 
        #Quantization for learned Embedding
        d_x,quant_input = quantizer.midrise(in_sym, 8, 4)
        inputs = quant_input.int()

        monitor.EPOCH = epoch_cnt+1;
        monitor.show()

        outputs,reg = model(inputs)
        A_eq[::,epoch_cnt] = torch.argmax(softmax(outputs),axis=1).int();

        REG[::,epoch_cnt] = reg

    REG = torch.sum(REG)/(sym_cnt*EPOCHS)

    return A_eq, REG

def Eval(Init_Model_Path, 
         NN_type, EQ_type,
         BAUDRATE, L, D, wavelen, SPS, beta, BIAS,
         ebno_dB, modem, EQ_TAP_CNT,
         EPOCHS, BATCHSIZE,
         op_point):
    with torch.no_grad():
        t_st = time.time()

        modulation_order = modem.give_mod_order();
        modulation       = modem.give_modulation();
        gray             = modem.is_gray_encoding();
        IMDD             = modem.give_IMDD();
        LMQ              = modem.give_LMQ();
        device           = modem.give_device();
     
        model = torch.load(Init_Model_Path, map_location=torch.device(device));  #initialize model and copy to target device
        model.device = device

        one_hot_encoder = None

        monitor = Process_Screen.Screen(PATH = Init_Model_Path, 
                         BAUDRATE=BAUDRATE, L=L, D=D, wavelen=wavelen,
                         BAUDRATE_TOTAL=BAUDRATE, L_TOTAL=L, wavelen_TOTAL=wavelen,SNR_TOTAL=ebno_dB,
                         SPS=SPS,beta=beta, BIAS=BIAS,
                         modulation_order=modulation_order,modulation=modulation,gray=gray,IMDD=IMDD,LMQ=LMQ,
                         NN_type=NN_type,EQ_type=EQ_type)

        loss_fn = torch.nn.CrossEntropyLoss()

        Symbol_energy = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        Noise_energy  = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        SER           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        BER           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        REG           = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        errors        = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        BER_Var       = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size));
        BER_Varbuf    = np.zeros( (wavelen.size,BAUDRATE.size,L.size,ebno_dB.size,EPOCHS));

        monitor.TOTAL_MAX=EPOCHS*BATCHSIZE*wavelen.size*BAUDRATE.size*L.size*ebno_dB.size

        for idx_Wl, Wl_i in enumerate(wavelen):
            monitor.wavelen = Wl_i;
            for idx_B, B_i in enumerate(BAUDRATE):
                monitor.B = B_i
                for idx_L, L_i in enumerate(L):
                    monitor.L = L_i
                    for idx,snr_i in enumerate(ebno_dB):
                        monitor.BATCHES = 1
                        monitor.BATCH   = 1;
                        monitor.SNR     = snr_i

                        if(op_point == None):
                            path = Init_Model_Path[:-3]+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(snr_i)+"_dB.sd"
                        else:
                            path = Init_Model_Path[:-3]+"_"+str(Wl_i)+"_nm_"+str(B_i)+"_Bd_"+str(L_i)+"_m_"+str(op_point)+"_dB.sd"

                        monitor.Init_Model_Path = path
                        model.load_state_dict(torch.load(path, map_location=torch.device(device)));

                        #Decomment this to check quantized weight performance
                        #import quantizer            #requires quantizer.py
                        ##w = 1
                        #w = 6
                        ##w = 4
                        ##w = 8
                        ##w = 16
                        #dx,quant = quantizer.midrise(model.encoding.weight.data,w,1);
                        #model.encoding.weight.data = dx*(quant-((2**w-1)/2)+0.5)
                        #model.encoding.weight.data = model.encoding.weight.detach()/torch.max(torch.abs(model.encoding.weight.detach()),1).values.reshape((-1,1)).repeat(1,model.encoding.weight.detach().shape[1])

                        batch_size_per_epoch = BATCHSIZE                                     #Number of symbols per batch
                        batch_size_per_epoch = batch_size_per_epoch//2 * 2                   #Ensure all batch sizes are even

                        e_st = time.time()

                        sym_cnt = int(batch_size_per_epoch)                             #Number of symbols
                        N       = modulation_order*sym_cnt                              #Number of bits
                        monitor.Batchsize = sym_cnt

                        monitor.TOTAL_IDX=(1+idx)*sym_cnt

                        channel = im_dd.IM_DD_Channel(sym_cnt,B_i,modulation_order,SPS,D,L_i,Wl_i,beta,snr_i,BIAS,device);

                        ##########################################################################
                        #   Create "iterations"-times parallel independent datasteams to create  #
                        #   to allow for parallel computation.                                   #
                        #   This is needed since due to the feedback an independent datasteam is #
                        #   not paralizable over the time domain, since the decisions of the     #
                        #   succeeding symbols depend on the previous decision.                  #
                        #   Allocate space to store the decisions which are fed back and         #
                        #   initialize them with the "preamble" aka previous symbols, since      #
                        #   the channel applies cyclic convolution.                              #
                        ##########################################################################

                        bits            = torch.zeros( (N      ,EPOCHS), device=device);
                        recv_bits       = torch.zeros( (N      ,EPOCHS), device=device);
                        A               = torch.zeros( (sym_cnt,EPOCHS), device=device);
                        A_eq            = torch.zeros( (sym_cnt,EPOCHS), device=device);
                        symbols         = torch.zeros( (sym_cnt,EPOCHS), device=device);
                        recv_symbols    = torch.zeros( (sym_cnt,EPOCHS), device=device);

                        for epoch_cnt in range(EPOCHS):
                            ##################################### Bits ################################################

                            bits[::,epoch_cnt] = torch.randint(0,2,size=(N,1),device=device).flatten();

                            ##################################### Sender ################################################

                            A[::,epoch_cnt],symbols[::,epoch_cnt] = modem.modulate(bits[::,epoch_cnt]);
                            symbols[::,epoch_cnt] = symbols[::,epoch_cnt] * np.sqrt(5)
                            rate = modulation_order;
                            esn0_dB = snr_i + 10 * np.log10( rate )

                            #################################### Channel  ################################################

                            recv_symbols[::,epoch_cnt] = channel.apply(symbols[::,epoch_cnt]);
                            recv_symbols[::,epoch_cnt] = recv_symbols[::,epoch_cnt] - torch.mean(recv_symbols[::,epoch_cnt]);
                            e_sym,e_noise = channel.give_energies();

                            Symbol_energy[idx_Wl][idx_B][idx_L][idx] += e_sym;
                            Noise_energy[idx_Wl][idx_B][idx_L][idx] += e_noise;

                        #Perform operations parallel
                        A_eq, reg = Operate_SNN_MMSE(model,recv_symbols, EQ_TAP_CNT, monitor, device);
                        
                        for epoch_cnt in range(EPOCHS):
                            delta_x, recv_bits[::,epoch_cnt] = modem.A_to_bits(A_eq[::,epoch_cnt]);

                        test                                    = (torch.sum(torch.abs(recv_bits-bits),axis=0)).detach().cpu()
                        BER[idx_Wl][idx_B][idx_L][idx]          = (torch.sum(test)/(bits.shape[0]*bits.shape[1])).numpy()
                        BER_Varbuf[idx_Wl][idx_B][idx_L][idx]   = (test/bits.shape[0]).numpy()
                        errors[idx_Wl][idx_B][idx_L][idx]       = (torch.sum(test)).numpy()
                        REG[idx_Wl][idx_B][idx_L][idx]          = reg.detach().cpu().numpy()

                        BER_Var[idx_Wl][idx_B][idx_L][idx] = 1/(EPOCHS-1)*np.sum(BER_Varbuf[idx_Wl][idx_B][idx_L][idx]**2)-EPOCHS/(EPOCHS-1)*BER[idx_Wl][idx_B][idx_L][idx]**2

                        monitor.REG = reg

                        classerr = torch.sum(torch.where(A_eq == A,0,1))
                        monitor.SER = classerr/(A.shape[0]*A.shape[1])*100
                        SER[idx_Wl][idx_B][idx_L][idx] = (classerr/(A.shape[0]*A.shape[1])).detach().cpu().numpy()

                        monitor.BER = BER[idx_Wl][idx_B][idx_L][idx] * 100
                            
                        e_et = time.time()
                        monitor.Time_per_Epoch = e_et-e_st;
                        
                        monitor.show()

                        del channel

        t_et = time.time()
        print()
        print("Time to eval: "+str(t_et-t_st)+"\n")

        #Write results to file
        if(op_point == None):
            fp = open("Evaluation_"+str(NN_type)+"_"+str(EQ_type)+".txt", 'w')
        else:
            fp = open("Evaluation_"+str(NN_type)+"_"+str(EQ_type)+"_Operation_Point_"+str(op_point)+"_dB.txt", 'w')
        fp.write("EbN0: "+str(ebno_dB)+"\n")
        fp.write("BER: "+str(BER)+"\n")
        fp.write("Errors: "+str(errors)+"\n")
        fp.write("SER: "+str(SER)+"\n")
        fp.write("Noisepower: "+str(-10*np.log10(Noise_energy/(SPS*BATCHSIZE*EPOCHS)))+"\n")
        fp.write("SNR: "+str(10*np.log10(Symbol_energy/(Noise_energy)))+"\n")
        fp.write("BER Variance: "+str(BER_Var)+"\n")
        fp.write("Regularization: "+str(REG)+"\n")
        fp.write("Time to eval: "+str(t_et-t_st)+"\n")
        fp.close()


