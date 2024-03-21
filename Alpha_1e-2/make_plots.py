import numpy as np
import matplotlib.pyplot as plt

sigma = np.arange(15,25)

Logscale = np.array([2.22693495e-02, 1.37985498e-02, 7.99554959e-03, 4.28130012e-03,
                     2.07949989e-03, 9.58000019e-04, 4.12199995e-04, 1.72300002e-04,
                     7.62499985e-05, 3.70999987e-05]);

Ternary = np.array([2.21117493e-02, 1.31326001e-02, 7.08679995e-03, 3.43684992e-03,
                    1.48854998e-03, 5.61150024e-04, 1.90349994e-04, 6.09999988e-05,
                    1.77000002e-05, 6.75000001e-06])

Alpha_1e2 = np.array([0.02530635, 0.01662595, 0.0104038 , 0.006257  , 
                      0.00359765, 0.00200915, 0.00112475, 0.00064095, 
                      0.00039565, 0.00026545] ) 

BER = np.array( [0.0268116 , 0.0177022 , 0.0111496 , 0.00676   , 
                 0.00392455, 0.00222185, 0.00124825, 0.00070285,
                 0.0004117 , 0.0002662 ] )
BER_2 = np.array([0.0308945 , 0.0210152 , 0.0138164 , 0.0087364 , 
                  0.0053924 , 0.0033066 , 0.00195875, 0.00116355, 
                  0.00074325, 0.00046895] )
BER_3 = np.array([0.02517795, 0.0164361 , 0.0101756 , 0.0059908 , 
                  0.00336365, 0.00182405, 0.00098495, 0.00055145, 
                  0.00032945, 0.00021605] )


plt.figure()
plt.semilogy(sigma,Logscale , label='Logscale',marker='o')

plt.semilogy(sigma,Ternary, label='Ternary',marker=9)

plt.semilogy(sigma,Alpha_1e2, label='Alpha 1e-2 (values from paper)',marker='x')

plt.semilogy(sigma,BER, label='Alpha 1e-2 (Addidional retraining and eval 1)',marker='*')
plt.semilogy(sigma,BER_2, label='Alpha 1e-2 (Addidional retraining and eval 2)',marker='*')
plt.semilogy(sigma,BER_3, label='Alpha 1e-2 (Addidional retraining and eval 3)',marker='*')


plt.title("4-PAM BER/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("BER");
plt.grid(which='both')
plt.legend();


Logscale     = np.ones(sigma.shape)*14.5/100
Ternary      = np.ones(sigma.shape)*12.5/100
Alpha_1e2    = np.ones(sigma.shape)* 5.2/100


Regularization = np.array( [0.04684418, 0.04698432, 0.0470895, 0.04718661, 
                            0.04724569, 0.04729539, 0.0473484, 0.04737619, 
                            0.04740594, 0.04743295] )

Regularization_2 = np.array( [0.05012426, 0.05036895, 0.05059098, 0.05075656, 
                              0.0509105 , 0.05104391, 0.05112018, 0.05119093, 
                              0.05125953, 0.05130028] )

Regularization_3 = np.array( [0.03910238, 0.03909527, 0.03908058, 0.03907624, 
                              0.03902314, 0.03898181, 0.03895052, 0.03891128, 
                              0.0389137 , 0.03886278] )

plt.figure()

plt.plot(sigma,Logscale, label='Logscale',marker='o')
plt.plot(sigma,Ternary, label='Ternary',marker=9)
plt.plot(sigma,Alpha_1e2, label='Alpha 1e-2 (values from paper)',marker='x')

plt.plot(sigma,Regularization, label='Alpha 1e-2 (Retraining 1)',marker='+')
plt.plot(sigma,Regularization_2, label='Alpha 1e-2 (Retraining 2)',marker='+')
plt.plot(sigma,Regularization_3, label='Alpha 1e-2 (Retraining 3)',marker='+')

plt.title("4-PAM Spikerate/Noisepower plot im/dd channel MMSE");
plt.xlabel("$-10log_{10}(\sigma^2)$");
plt.ylabel("Spikerate");
plt.grid(which='both')
plt.legend();

plt.show()

