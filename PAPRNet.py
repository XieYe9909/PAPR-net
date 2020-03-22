import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Lambda
from keras.backend import mean, var, max, abs, square, sqrt


def float2cmplx(input_layer):
    return tf.complex(input_layer[:, 0:1, :], input_layer[:, 1:2, :])


def cmplx2float(input_layer):
    return tf.concat((tf.math.real(input_layer), tf.math.imag(input_layer)), axis=-2)


def tf_hermitian(input_layer):
    a = tf.zeros(tf.shape(input_layer[:, :, 0:1]), dtype=tf.complex64)
    input_layer_conj = tf.math.conj(input_layer[:, :, ::-1])
    return tf.concat((a, input_layer, a, input_layer_conj), axis=-1)


def tf_inv_hermitian(input_layer):
    n = int(input_layer.shape[-1] / 2)
    return input_layer[:, :, 1: n]


def add_noise(input_layer, snr):
    sig_power = mean(square(abs(input_layer)))
    noise_power = sig_power * (10 ** (-snr / 10)) / 2
    noise_real = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=sqrt(noise_power), dtype=tf.float32)
    noise_imag = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=sqrt(noise_power), dtype=tf.float32)
    noise = tf.complex(noise_real, noise_imag)
    return input_layer + noise


def tf_papr(_, sig):
    sig_power = var(sig, axis=-1)
    sig_max = square(max(abs(sig), axis=-1))
    return sig_max / sig_power


def papr_encoder(in_shape: tuple, n_hidden1: int, n_hidden2: int, n_out: int) -> Model:
    model = Sequential(name='encoder')

    model.add(Dense(n_hidden1, input_shape=in_shape, activation='relu', name='hidden1'))
    model.add(BatchNormalization(name='BN1'))

    model.add(Dense(n_hidden2, activation='relu', name='hidden2'))
    model.add(BatchNormalization(name='BN2'))

    model.add(Dense(n_out, name='enc_out'))

    return model


def papr_decoder(in_shape: tuple, n_hidden1: int, n_hidden2: int, n_out: int) -> Model:
    model = Sequential(name='decoder')

    model.add(Dense(n_hidden1, input_shape=in_shape, activation='relu', name='hidden3'))
    model.add(BatchNormalization(name='BN3'))

    model.add(Dense(n_hidden2, activation='relu', name='hidden4'))
    model.add(BatchNormalization(name='BN4'))

    model.add(Dense(n_out, name='dec_out'))

    return model


def papr_net(enc: Model, dec: Model, channel=None, snr=None) -> Model:
    enc_in = enc.input
    enc_out = enc(enc_in)

    ofdm_chunk = Lambda(float2cmplx, name='float2cmplx')(enc_out)
    ofdm_chunk_herm = Lambda(tf_hermitian, name='hermitian')(ofdm_chunk)
    sig = Lambda(tf.signal.ifft, name='ifft')(ofdm_chunk_herm)
    sig = Lambda(tf.math.real, name='signal')(sig)

    if snr:
        ofdm_chunk = Lambda(lambda x: tf.multiply(x, channel), name='add_channel')(ofdm_chunk)
        ofdm_chunk = Lambda(lambda x: add_noise(x, snr), name='awgn')(ofdm_chunk)
        ofdm_chunk = Lambda(lambda x: tf.divide(x, channel), name='de_channel')(ofdm_chunk)

        ofdm_chunk = Lambda(cmplx2float, name='cmplx2float')(ofdm_chunk)
        dec_out = dec(ofdm_chunk)
    else:
        dec_out = dec(enc_out)

    return Model(inputs=enc_in, outputs=[dec_out, sig], name='papr_net')
