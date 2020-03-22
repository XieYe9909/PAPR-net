import numpy as np
from utils import bin2dec, dec2bin
from scipy.io import loadmat
from keras.models import load_model

qam_map = loadmat('initialize/qam_map.mat')["qam_map"]


def hermitian(input_array: np.ndarray) -> np.ndarray:
    if np.ndim(input_array) == 1:
        input_array = np.reshape(input_array, (1, -1))
    shape = np.shape(input_array)
    output_array = np.zeros((shape[0], 2 * shape[1]), dtype=np.complex)
    output_array[:, 0] = np.real(input_array[:, -1])
    output_array[:, shape[1]] = np.imag(input_array[:, -1])
    output_array[:, 1: shape[1]] = input_array[:, 0: -1]
    output_array[:, shape[1] + 1:] = np.conjugate(input_array[:, 0: -1])[:, ::-1]
    return output_array


def inv_hermitian(input_array: np.ndarray) -> np.ndarray:
    if np.ndim(input_array) == 1:
        input_array = np.reshape(input_array, (1, -1))
    shape = np.shape(input_array)
    output_array = np.zeros((shape[0], int(shape[1] / 2)), dtype=np.complex)
    output_array[:, 0: -1] = input_array[:, 1: int(shape[1] / 2)]
    output_array[:, -1] = input_array[:, 0] + 1j * input_array[:, int(shape[1] / 2)]
    return output_array


def oversampling(ofdm_herm: np.ndarray, os_rate: int) -> np.ndarray:
    size = np.shape(ofdm_herm)
    subcarrier_num = int(size[1] / 2)
    if os_rate == 1:
        ofdm_herm_os = ofdm_herm
    else:
        ofdm_herm_os = os_rate * np.hstack((ofdm_herm[:, 0: subcarrier_num + 1],
                                            np.zeros((size[0], (os_rate - 1) * size[1] - 1), dtype=np.complex),
                                            (ofdm_herm[:, subcarrier_num:])))
    return ofdm_herm_os


def qammod(dec_val: int, qam_num: int) -> complex:
    if dec_val >= 2 ** qam_num:
        pass
    else:
        return qam_map[qam_num - 1, dec_val]


def qamdemod(complex_val: complex, qam_num: int):
    d = np.abs(qam_map[qam_num - 1, 0: 2 ** qam_num] - complex_val)
    return np.argmin(d).astype(np.int)


class OFDMTransmitter(object):

    def __init__(self, **params: dict):
        self.BitAllocation = params['BitAllocation']    # 比特分配方案
        self.Coef = params['Coef']  # 功率归一化系数
        self.OverSampleRate = params['OverSampleRate'] if 'OverSampleRate' in params else 1  # 过采样率
        self.CPLength = params['CPLength'] if 'CPLength' in params else 0  # 循环前缀长度
        self.SubCarrierNum = len(self.BitAllocation)    # 子信道个数

    def get_qammod(self, bit_stream):
        bits_per_symbol = sum(self.BitAllocation)
        symbol_num = int(len(bit_stream) / bits_per_symbol)
        bit_stream = bit_stream.reshape((symbol_num, bits_per_symbol))

        ofdm_chunks = np.zeros((symbol_num, self.SubCarrierNum), dtype=np.complex)
        for i in range(symbol_num):
            k = 0
            for j in range(self.SubCarrierNum):
                if self.BitAllocation[j] > 0:
                    dec_val = bin2dec(bit_stream[i, k: k + self.BitAllocation[j]], 'left-msb')
                    ofdm_chunks[i, j] = qammod(dec_val, self.BitAllocation[j])
                    k += self.BitAllocation[j]

        return ofdm_chunks

    def transmit(self, bit_stream, encoder=None, papr_net=False):
        ofdm_chunks = self.get_qammod(bit_stream)

        if papr_net:
            ofdm_chunks = np.concatenate([np.expand_dims(ofdm_chunks.real, 1),
                                          np.expand_dims(ofdm_chunks.imag, 1)], axis=1)
            enc = load_model(encoder)
            ofdm_chunks[:, :, 0: -1] = enc.predict(ofdm_chunks[:, :, 0: -1])
            ofdm_chunks = ofdm_chunks[:, 0, :] + 1j * ofdm_chunks[:, 1, :]
        else:
            ofdm_chunks *= self.Coef

        ofdm_chunks_herm = hermitian(ofdm_chunks)
        ofdm_chunks_herm_os = oversampling(ofdm_chunks_herm, self.OverSampleRate)

        sig = np.real(np.fft.ifft(ofdm_chunks_herm_os))
        cyclic_prefix = sig[:, -self.CPLength:]
        sig = np.hstack((cyclic_prefix, sig))
        sig = np.reshape(sig, (-1,))

        return sig


class OFDMReceiver(object):

    def __init__(self, **params):
        self.BitAllocation = params['BitAllocation']  # 比特分配方案
        self.Coef = params['Coef']  # 功率归一化系数
        self.OverSampleRate = params['OverSampleRate']  # 过采样率
        self.CPLength = params['CPLength']  # 循环前缀长度
        self.SubCarrierNum = len(self.BitAllocation)  # 子信道个数

    def get_qammod(self, sig: np.ndarray, shift: int):
        ofdm_len = 2 * self.OverSampleRate * self.SubCarrierNum
        total_len = ofdm_len + self.CPLength
        sym_num = int(sig.size / total_len)

        sig = sig.reshape((sym_num, total_len))
        sig = sig[:, shift: shift + ofdm_len]

        ofdm_chunks_herm_os = np.fft.fft(sig)
        if self.OverSampleRate == 1:
            ofdm_chunks_herm = ofdm_chunks_herm_os
        else:
            ofdm_chunks_herm = np.hstack((ofdm_chunks_herm_os[:, 0: self.SubCarrierNum],
                                          ofdm_chunks_herm_os[:, -self.SubCarrierNum:])) / self.OverSampleRate

        ofdm_chunks = inv_hermitian(ofdm_chunks_herm)
        return ofdm_chunks

    def get_recv_qam(self, sig: np.ndarray, shift: int, channel: np.ndarray, decoder=None, papr_net=False):
        ofdm_chunks = self.get_qammod(sig, shift)
        ofdm_chunks /= channel

        if papr_net:
            ofdm_chunks = np.concatenate([np.expand_dims(ofdm_chunks.real, 1),
                                          np.expand_dims(ofdm_chunks.imag, 1)], axis=1)
            dec = load_model(decoder)
            ofdm_chunks[:, :, 0: -1] = dec.predict(ofdm_chunks[:, :, 0: -1])
            ofdm_chunks = ofdm_chunks[:, 0, :] + 1j * ofdm_chunks[:, 1, :]
        else:
            ofdm_chunks /= self.Coef

        return ofdm_chunks

    def receive(self, sig: np.ndarray, shift: int, channel: np.ndarray, decoder=None, papr_net=False):
        ofdm_chunks = self.get_recv_qam(sig, shift=shift, channel=channel, decoder=decoder, papr_net=papr_net)

        sym_num = ofdm_chunks.shape[0]
        bits_per_symbol = sum(self.BitAllocation)

        bit_stream = np.zeros((sym_num, bits_per_symbol), dtype=np.int)
        for i in range(sym_num):
            k = 0
            for j in range(self.SubCarrierNum):
                if self.BitAllocation[j] > 0:
                    dec_val = qamdemod(ofdm_chunks[i, j], self.BitAllocation[j])
                    bit_stream[i, k: k + self.BitAllocation[j]] = dec2bin(dec_val, self.BitAllocation[j], 'left-msb')
                    k += self.BitAllocation[j]
        bit_stream = bit_stream.reshape((-1,))

        return bit_stream
