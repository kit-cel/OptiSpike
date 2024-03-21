import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

Logscale = np.array([2.22693495e-02, 1.37985498e-02, 7.99554959e-03, 4.28130012e-03,
                     2.07949989e-03, 9.58000019e-04, 4.12199995e-04, 1.72300002e-04,
                     7.62499985e-05, 3.70999987e-05]);

Ternary = np.array([2.21117493e-02, 1.31326001e-02, 7.08679995e-03, 3.43684992e-03,
                    1.48854998e-03, 5.61150024e-04, 1.90349994e-04, 6.09999988e-05,
                    1.77000002e-05, 6.75000001e-06])

Alpha_1e9 = np.array([0.0191027503  , 0.0111319497  , 0.00580235012 , 0.00270275003 ,
                      0.00111920002 , 0.000415199989, 0.000130050001, 3.84999985e-05, 
                      1.02499998e-05, 3.24999996e-06] )

BER = np.array( [1.92339495e-02, 1.11619504e-02, 5.85740013e-03, 2.79349997e-03,
                 1.16180000e-03, 4.35099995e-04, 1.42399993e-04, 4.55500012e-05,
                 1.40000002e-05, 5.84999998e-06] )

BER_2 = np.array( [1.99242495e-02, 1.17929000e-02, 6.32114988e-03, 3.07490001e-03,
                   1.32489996e-03, 5.26249991e-04, 1.84849996e-04, 6.44999964e-05,
                   2.33999999e-05, 9.89999990e-06] )

BER_3 = np.array( [1.91758499e-02, 1.11122997e-02, 5.81840007e-03, 2.71894992e-03,
                   1.13454997e-03, 4.01600002e-04, 1.32500005e-04, 4.00499994e-05,
                   1.34499996e-05, 3.65000005e-06] )
BER_W16 = np.array([1.91488005e-02, 1.11344000e-02, 5.80684980e-03, 2.69469991e-03,
                    1.14199996e-03, 4.13650007e-04, 1.34799993e-04, 3.99999990e-05,
                    1.28000001e-05, 4.64999994e-06])
BER_W8 = np.array([1.93353500e-02, 1.12244999e-02, 5.93770016e-03, 2.81274994e-03,
                   1.18499994e-03, 4.30349988e-04, 1.46549995e-04, 4.26500010e-05,
                   1.54999998e-05, 4.74999979e-06])
BER_W6 = np.array([2.14770995e-02, 1.31195001e-02, 7.31194997e-03, 3.76845011e-03,
                   1.75439997e-03, 7.39049981e-04, 3.08599992e-04, 1.15050003e-04,
                   4.42499986e-05, 1.89500006e-05])
BER_W4 = np.array([0.07971295, 0.0703941 , 0.0627727,  0.0561618 , 
                   0.0507697 , 0.0463808 , 0.0424057,  0.03990895, 
                   0.0371745 , 0.03534695] )

plt.figure()

plt.semilogy(sigma,Logscale , label='Logscale',marker='o')

plt.semilogy(sigma,Ternary, label='Ternary',marker=9)

plt.semilogy(sigma, Alpha_1e9 , label='Alpha 1e-9 (values from paper)',marker='x')

plt.semilogy(sigma, BER , label='Alpha 1e-9 (Retraining and eval 1)',marker='x')
plt.semilogy(sigma, BER_2 , label='Alpha 1e-9 (Retraining and eval 2)',marker='x')
plt.semilogy(sigma, BER_3 , label='Alpha 1e-9 (Retraining and eval 3)',marker='x')
plt.semilogy(sigma, BER_W16 , label='Alpha 1e-9 (Retraining and eval 3) 16Bits Quantized Weights',marker='x')
plt.semilogy(sigma, BER_W8 , label='Alpha 1e-9 (Retraining and eval 3) 8Bits Quantized Weights',marker='x')
plt.semilogy(sigma, BER_W6 , label='Alpha 1e-9 (Retraining and eval 3) 6Bits Quantized Weights',marker='x')
plt.semilogy(sigma, BER_W4 , label='Alpha 1e-9 (Retraining and eval 3) 4Bits Quantized Weights',marker='x')


plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();




Logscale     = np.ones(sigma.shape)*14.5/100
Ternary      = np.ones(sigma.shape)*12.5/100
Alpha_1e9    = np.ones(sigma.shape)* 7.3/100

Regularization = np.array( [0.06612079, 0.06619126, 0.0662796 , 0.06636735, 
                            0.06646375, 0.06653123, 0.06661432, 0.06665206, 
                            0.0667075 , 0.0667505 ])

Regularization_2 = np.array( [0.07080225, 0.0709552 , 0.07116978, 0.07128048, 
                              0.07143121, 0.07154017, 0.07163629, 0.07173959, 
                              0.07178911, 0.07183908] )
Regularization_3 = np.array( [0.07234412, 0.07258388, 0.07275919, 0.07292519, 
                              0.07307923, 0.07323492, 0.07333293, 0.07341547, 
                              0.07351007, 0.0735551 ] )

plt.figure()

plt.plot(sigma,Logscale, label='Logscale',marker='o')
plt.plot(sigma,Ternary, label='Ternary',marker=9)
plt.plot(sigma,Alpha_1e9, label='Alpha 1e-9 (values from paper)',marker='x')

plt.plot(sigma,Regularization, label='Alpha 1e-9 (Retraining 1)',marker='+')
plt.plot(sigma,Regularization_2, label='Alpha 1e-9 (Retraining 2)',marker='+')
plt.plot(sigma,Regularization_3, label='Alpha 1e-9 (Retraining 3)',marker='+')

plt.title("4-PAM Spikerate/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("Spikerate");
plt.grid(which='both')
plt.legend();

plt.show()

