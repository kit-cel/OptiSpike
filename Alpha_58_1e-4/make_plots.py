import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

Logscale = np.array([2.22693495e-02, 1.37985498e-02, 7.99554959e-03, 4.28130012e-03,
                     2.07949989e-03, 9.58000019e-04, 4.12199995e-04, 1.72300002e-04,
                     7.62499985e-05, 3.70999987e-05]);

Ternary = np.array([2.21117493e-02, 1.31326001e-02, 7.08679995e-03, 3.43684992e-03,
                    1.48854998e-03, 5.61150024e-04, 1.90349994e-04, 6.09999988e-05,
                    1.77000002e-05, 6.75000001e-06])

Alpha_58_1e4 = np.array([0.0226360504  , 0.013979      , 0.00808249973, 0.00430034986 ,
                         0.00211860007 , 0.000983149977, 0.00043700001, 0.000188899998, 
                         8.88499999e-05, 4.47999992e-05] ) 


BER = np.array( [2.21563503e-02, 1.36895999e-02, 7.85985030e-03, 4.16945014e-03,
                 2.03419989e-03, 9.34550015e-04, 4.00149991e-04, 1.80300005e-04,
                 8.52500016e-05, 4.47999992e-05] )
BER_2 = np.array( [2.38669496e-02, 1.52513003e-02, 8.94694962e-03, 4.94029978e-03,
                   2.54944991e-03, 1.22510002e-03, 5.70899982e-04, 2.53600010e-04,
                   1.21049998e-04, 6.57999990e-05] )
BER_3 = np.array( [2.15557497e-02, 1.32762501e-02, 7.53004989e-03, 3.84570006e-03,
                   1.84489996e-03, 8.06549971e-04, 3.29350005e-04, 1.34450005e-04,
                   6.02500004e-05, 2.65500003e-05] )


plt.figure()

plt.semilogy(sigma,Logscale , label='Logscale',marker='o')

plt.semilogy(sigma,Ternary, label='Ternary',marker=9)

plt.semilogy(sigma, Alpha_58_1e4 , label='Alpha 5.8*1e-4 (values from paper)',marker='x')

plt.semilogy(sigma, BER , label='Alpha 5.8*1e-4 (Retraining and eval 1)',marker='x')
plt.semilogy(sigma, BER_2 , label='Alpha 5.8*1e-4 (Retraining and eval 2)',marker='x')
plt.semilogy(sigma, BER_3 , label='Alpha 5.8*1e-4 (Retraining and eval 3)',marker='x')


plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();



Logscale     = np.ones(sigma.shape)*14.5/100
Ternary      = np.ones(sigma.shape)*12.5/100
Alpha_58_1e4 = np.ones(sigma.shape)* 6.0/100

Regularization = np.array( [0.06396945, 0.06424044, 0.0645055 , 0.06473609, 
                            0.06492099, 0.06507629, 0.06520271, 0.06531925, 
                            0.06539541, 0.06546944] )
Regularization_2 = np.array([0.06471326, 0.06502184, 0.06532276, 0.06557834, 
                             0.06579918, 0.06597143, 0.06611181, 0.06623268, 
                             0.06634028, 0.06644161] )
Regularization_3 = np.array([0.06382381, 0.0642024 , 0.0645201 , 0.0648203 , 
                             0.06503088, 0.06523495, 0.06539746, 0.06553219, 
                             0.06563181, 0.06569914] )

plt.figure()

plt.plot(sigma,Logscale, label='Logscale',marker='o')

plt.plot(sigma,Ternary, label='Ternary',marker=9)

plt.plot(sigma,Alpha_58_1e4, label='Alpha 5.8*1e-4 (values from paper)',marker='x')

plt.plot(sigma,Regularization, label='Alpha 5.8*1e-4 (Retraining 1)',marker='+')
plt.plot(sigma,Regularization_2, label='Alpha 5.8*1e-4 (Retraining 2)',marker='+')
plt.plot(sigma,Regularization_3, label='Alpha 5.8*1e-4 (Retraining 3)',marker='+')

plt.title("4-PAM Spikerate/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("Spikerate");
plt.grid(which='both')
plt.legend();

plt.show()

