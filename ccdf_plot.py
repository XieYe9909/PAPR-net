import numpy as np
import matplotlib.pyplot as plt
from ofdm import OFDMTransmitter
from signals import papr_cdf
from initialize.params import bit_allocation, coef

params = dict()
params['BitAllocation'] = bit_allocation
params['Coef'] = coef
params['OverSampleRate'] = 4
params['CPLength'] = 100

symbol_num = 5000
bits_per_symbol = sum(params['BitAllocation'])
bit_send = np.random.randint(0, 2, bits_per_symbol * symbol_num)

encoder_list = [
    None,
    'training/encoder_3.hdf5',
    'training/encoder_5.hdf5',
    'training/encoder_10.hdf5',
    'training/encoder_30.hdf5'
]

tx = OFDMTransmitter(**params)
sig_list = list()
for encoder in encoder_list:
    if encoder:
        sig = tx.transmit(bit_send, encoder=encoder, papr_net=True)
    else:
        sig = tx.transmit(bit_send, papr_net=False)

    sig = sig.reshape((symbol_num, -1))
    sig_list.append(sig)

start = 6
end = 15
step = 0.1
n = np.arange(start, end + step, step)
sig_cdf = np.zeros((len(sig_list), len(n)))

for [i, sig] in enumerate(sig_list):
    [_, sig_cdf[i]] = papr_cdf(sig, start, end, step)

plt.figure()
plt.plot(n, sig_cdf[0], label='conventional OFDM', color='b', marker='o')
plt.plot(n, sig_cdf[1], label=r'PAPR net $ \lambda $ = 0.03', color='r', marker='o')
plt.plot(n, sig_cdf[2], label=r'PAPR net $ \lambda $ = 0.05', color='g', marker='o')
plt.plot(n, sig_cdf[3], label=r'PAPR net $ \lambda $ = 0.1', color='c', marker='o')
plt.plot(n, sig_cdf[4], label=r'PAPR net $ \lambda $ = 0.3', color='m', marker='o')

plt.xlabel('PAPR')
plt.ylabel('CCDF')
plt.semilogy()
plt.grid(True)
plt.legend(loc=1)
plt.savefig('results/ccdf.eps', dpi=600, format='eps')
plt.show()
