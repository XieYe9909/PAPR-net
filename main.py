import numpy as np
import matplotlib.pyplot as plt
from ofdm import OFDMTransmitter, OFDMReceiver
from signals import papr_calc, awgn, ber_subcarrier_calc
from initialize.params import channel, h, bit_allocation, coef

params = dict()
params['BitAllocation'] = bit_allocation
params['Coef'] = coef
params['OverSampleRate'] = 4
params['CPLength'] = 100

symbol_num = 1000
bits_per_symbol = np.sum(params['BitAllocation'])
bit_send = np.random.randint(0, 2, bits_per_symbol * symbol_num)

tx = OFDMTransmitter(**params)
sig = tx.transmit(bit_send, encoder='training/encoder_30.hdf5', papr_net=True)
papr = papr_calc(sig)

plt.figure(1)
plt.plot(sig[0: 10 * 2148])
plt.show()

sig_recv = np.convolve(sig, h)[0: len(sig)]
sig_recv = awgn(sig_recv, snr=20)

rx = OFDMReceiver(**params)
bit_recv = rx.receive(sig_recv, shift=rx.CPLength, channel=channel,
                      decoder='training/decoder_30.hdf5', papr_net=True)

ber = np.sum(np.logical_xor(bit_send, bit_recv)) / bit_send.size
print('PAPR = %.4f\nBER = %.6f' % (papr, ber))

ber_subcarrier = ber_subcarrier_calc(bit_send, bit_recv, bit_allocation)

plt.figure(2)
plt.plot(ber_subcarrier, marker='o')
plt.show()


# rx = OFDMReceiver(**params)
# ofdm_chunks = rx.get_recv_qam(sig_recv, shift=rx.CPLength, channel=channel,
#                               decoder='training/decoder_30.hdf5', papr_net=True)
#
# k = 0
# plt.figure(1)
# plt.scatter(ofdm_chunks[:, k].real, ofdm_chunks[:, k].imag)
# plt.show()
