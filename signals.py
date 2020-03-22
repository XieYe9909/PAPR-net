import numpy as np
from math import floor


def papr_calc(sig):
    sig_power = np.var(sig)
    sig_max = np.max(np.abs(sig)) ** 2
    papr = 10 * np.log10(sig_max / sig_power)
    return papr


def cdf_calc(array, start, end, step):
    xrange = np.arange(start, end + step, step)
    length = xrange.size

    cdf = np.zeros(xrange.shape)
    for x in array:
        i = floor((x - start) / step)
        if i >= length:
            cdf[-1] += 1
        elif i >= 0:
            cdf[i] += 1
        else:
            continue

    for i in range(length - 2, -1, -1):
        cdf[i] += cdf[i + 1]

    cdf /= len(array)

    return [xrange, cdf]


def papr_cdf(sig_array, start, end, step):
    sig_num = sig_array.shape[0]
    papr_array = np.zeros(sig_num)

    for i in range(sig_num):
        papr_array[i] = papr_calc(sig_array[i])

    return cdf_calc(papr_array, start, end, step)


def awgn(sig: np.ndarray, snr: float, sig_power=None):
    if sig_power is None:
        sig_power = np.mean(np.square(sig))

    noise_power = sig_power * (10 ** (-snr / 10))
    noise = np.random.normal(loc=0, scale=np.sqrt(noise_power), size=sig.shape)
    return sig + noise


def ber_subcarrier_calc(bit_send, bit_recv, bit_allocation: np.ndarray):
    bits_per_symbol = bit_allocation.sum()

    bit_send = bit_send.reshape((-1, bits_per_symbol))
    bit_recv = bit_recv.reshape((-1, bits_per_symbol))
    ber_subcarrier = np.zeros(bit_allocation.size)

    k = 0
    for j, bit_num in enumerate(bit_allocation):
        if bit_num > 0:
            send = bit_send[:, k: k + bit_num]
            recv = bit_recv[:, k: k + bit_num]
            ber_subcarrier[j] = np.sum(np.logical_xor(send, recv)) / send.size
            k += bit_num

    return ber_subcarrier
