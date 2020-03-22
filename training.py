import numpy as np
from ofdm import OFDMTransmitter
from PAPRNet import papr_net, tf_papr
from initialize.params import bit_allocation, coef, channel
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import adam
from keras.models import load_model

params = dict()
params['BitAllocation'] = bit_allocation
params['Coef'] = coef

train_num = 5000
test_num = 100
symbol_num = train_num + test_num
bits_per_symbol = sum(params['BitAllocation'])
bit_send = np.random.randint(0, 2, bits_per_symbol * symbol_num)

tx = OFDMTransmitter(**params)
ofdm_chunks = tx.get_qammod(bit_send)
ofdm_chunks = np.concatenate([np.expand_dims(ofdm_chunks.real, 1), np.expand_dims(ofdm_chunks.imag, 1)], axis=1)

ofdm_train = ofdm_chunks[0: train_num, :, 0: -1]
ofdm_train = np.repeat(ofdm_train, repeats=6, axis=0)
ofdm_test = ofdm_chunks[train_num:, :, 0: -1]
sig_train = np.zeros(6 * train_num)
sig_test = np.zeros(test_num)

encoder = load_model('pre_train/encoder.hdf5')
decoder = load_model('pre_train/decoder.hdf5')

rate = 30
net = papr_net(encoder, decoder, channel=channel[0: -1], snr=20)

callbacks = list()
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=0.01,
                                   cooldown=0, min_lr=1e-10))
callbacks.append(CSVLogger('training/train_log_%d.csv' % rate, separator=',', append=False))
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=0, mode='min'))

optimizer = adam(learning_rate=0.001)
net.compile(loss=['mse', tf_papr], loss_weights=[1, rate / 100], optimizer=optimizer)
net.fit(ofdm_train, [ofdm_train, sig_train], validation_data=(ofdm_test, [ofdm_test, sig_test]), epochs=120,
        batch_size=50, callbacks=callbacks)

encoder.save('training/encoder_%d.hdf5' % rate)
decoder.save('training/decoder_%d.hdf5' % rate)
