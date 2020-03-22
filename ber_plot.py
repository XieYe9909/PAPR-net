import numpy as np
import matplotlib.pyplot as plt
from ofdm import OFDMTransmitter, OFDMReceiver
from initialize.params import channel, h, bit_allocation, coef
from signals import awgn

params = dict()
params['BitAllocation'] = bit_allocation
params['Coef'] = coef
params['OverSampleRate'] = 4
params['CPLength'] = 100

symbol_num = 1000
bits_per_symbol = sum(params['BitAllocation'])
bit_send = np.random.randint(0, 2, bits_per_symbol * symbol_num)

tx = OFDMTransmitter(**params)
rx = OFDMReceiver(**params)

encoder_list = [
    None,
    'training/encoder_3.hdf5',
    'training/encoder_5.hdf5',
    'training/encoder_10.hdf5',
    'training/encoder_30.hdf5',
]

decoder_list = [
    None,
    'training/decoder_3.hdf5',
    'training/decoder_5.hdf5',
    'training/decoder_10.hdf5',
    'training/decoder_30.hdf5',
]

sig_list = list()
for encoder in encoder_list:
    if encoder:
        sig = tx.transmit(bit_send, encoder=encoder, papr_net=True)
    else:
        sig = tx.transmit(bit_send, papr_net=False)

    sig_list.append(sig)

for [i, sig] in enumerate(sig_list):
    sig_list[i] = np.convolve(sig, h)[0: sig.size]

snr_list = np.arange(0, 32, 1)
ber_list = np.zeros((len(sig_list), len(snr_list)))

for [i, sig] in enumerate(sig_list):
    sig_power = np.var(sig)
    decoder = decoder_list[i]

    for j, snr in enumerate(snr_list):
        sig_n = awgn(sig, snr=snr, sig_power=sig_power)

        if decoder:
            bit_recv = rx.receive(sig_n, shift=rx.CPLength, channel=channel, decoder=decoder, papr_net=True)
        else:
            bit_recv = rx.receive(sig_n, shift=rx.CPLength, channel=channel, papr_net=False)

        ber_list[i, j] = np.sum(np.logical_xor(bit_send, bit_recv)) / bit_send.size
        print('%d' % j)

plt.figure()
plt.plot(snr_list, ber_list[0], label='conventional OFDM', color='b', marker='o')
plt.plot(snr_list, ber_list[1], label=r'PAPR net $ \lambda $ = 0.03', color='r', marker='o')
plt.plot(snr_list, ber_list[2], label=r'PAPR net $ \lambda $ = 0.05', color='g', marker='o')
plt.plot(snr_list, ber_list[3], label=r'PAPR net $ \lambda $ = 0.1', color='c', marker='o')
plt.plot(snr_list, ber_list[4], label=r'PAPR net $ \lambda $ = 0.3', color='m', marker='o')

plt.xlabel('snr (dB)')
plt.ylabel('BER')
plt.semilogy()
plt.legend()
plt.grid()
plt.savefig('results/ber.eps', dpi=600, format='eps')
plt.show()
