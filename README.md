# PAPR-net
Deep learning for PAPR reduction in OFDM system.

## Operating Environment
Python 3.7 + Keras based on Tensorflowbackend.

## Two Training Steps
For the networks training process, it can be divided into two steps.

**Step 1:** The parameters in encoder and decoder are initialized randomly. There is only BER performance in the loss fuction. Moreover, There is no channel model and AWGN model in the PAPR net. The output of the encoder is exactly the input of the decoder.

**Step 2:** Use the parameters trained in Step 1 as the initial values. The loss function is as below:
$$x_2$$


## Main Functions
The main usage of each file or folder is listed below: 

**ofdm.py:** Two main classes: OFDM transmitter and Receiver, used for transmitting and receiving bit streams. Some functions: sunch qam mapping, hermitian transformation and oversampling.

**PAPRNet.py:** Mainly about the definition of the encoder, decoder and the PAPR net, also some functions used in intermediate layers between the encoder and decoder.

**signals.py:** Some simple function for the time-domain OFDM signals, used for calculating the PAPR of a input signal (papr_calc), add white Gaussian noise to a signal (awgn), and calculating BER for each subcarrier (ber_subcarrier_calc), etc.

**utils.py:** Hex convertion. Binary to decimal (bin2dec) and decimal to binary (dec2bin).

**pre_train.py:** Step 1 in the training process.

**training.py:** Step 2 in the training process.

**main.py:** The entire process of transmitting and receiving bit streams, also calculating the PAPR for the time-domain signal and the BER for each subcarrier.

**ber_plot.py:** Drawing the BER vs. snr curve for different systems of different PAPR nets.

**ccdf_plot.py:** Drawing the CCDF curve for different systems of different PAPR nets.

**initialize/:** Saving some important parameters used in the OFDM system, such as the channel response model, the QAM map, etc.

**pre_train/:** Saving the training model after training step 1 (pre_train), which are used in training step 2.

**training/:** Saving the training model after training step 2.

**results/:** Saving the CCDF and BER results generated in ber_plot.py and ccdf_plot.py.
