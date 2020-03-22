import numpy as np


def bin2dec(bits, msb='left-msb'):
    dec_val = 0
    for i, j in enumerate(bits):
        if msb == 'left-msb':
            dec_val += j << (len(bits) - i - 1)
        elif msb == 'right-msb':
            dec_val += j << i
        else:
            raise TypeError('error')
    return dec_val


def dec2bin(val, width, msb='left-msb'):
    if val >= 2 ** width:
        pass
    if msb == 'left-msb':
        bits = np.binary_repr(val, width)
    elif msb == 'right-msb':
        bits = np.binary_repr(val, width)[::-1]
    else:
        raise TypeError('error')
    bits = ''.join(bits)
    bits = np.array([int(b) for b in bits])
    return bits
