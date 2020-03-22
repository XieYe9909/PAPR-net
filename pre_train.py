import numpy as np
from ofdm import OFDMTransmitter
from PAPRNet import papr_encoder, papr_decoder, papr_net, tf_papr
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import adam
from initialize.params import bit_allocation, coef

params = dict()
params['BitAllocation'] = bit_allocation
params['Coef'] = coef

train_num = 10000
test_num = 100
symbol_num = train_num + test_num
bits_per_symbol = sum(params['BitAllocation'])
bit_send = np.random.randint(0, 2, bits_per_symbol * symbol_num)

tx = OFDMTransmitter(**params)
ofdm_chunks = tx.get_qammod(bit_send)
ofdm_chunks = np.concatenate([np.expand_dims(ofdm_chunks.real, 1), np.expand_dims(ofdm_chunks.imag, 1)], axis=1)

ofdm_train = ofdm_chunks[0: train_num, :, 0: -1]
ofdm_test = ofdm_chunks[train_num:, :, 0: -1]
sig_train = np.zeros(train_num)
sig_test = np.zeros(test_num)

n = tx.SubCarrierNum
encoder = papr_encoder((2, n - 1), 2 * (n - 1), 2 * (n - 1), n - 1)
decoder = papr_decoder((2, n - 1), 2 * (n - 1), 2 * (n - 1), n - 1)
net = papr_net(encoder, decoder)
print(net.summary())

callbacks = list()
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='min', min_delta=0.01,
                                   cooldown=0, min_lr=1e-10))
callbacks.append(CSVLogger('pre_train/train_log.csv', separator=',', append=False))
callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, verbose=0, mode='min'))

optimizer = adam(learning_rate=0.001)
net.compile(loss=['mse', tf_papr], loss_weights=[1, 0], optimizer=optimizer)
net.fit(ofdm_train, [ofdm_train, sig_train], validation_data=(ofdm_test, [ofdm_test, sig_test]), epochs=100,
        batch_size=50, callbacks=callbacks)

encoder.save('pre_train/encoder.hdf5')
decoder.save('pre_train/decoder.hdf5')
