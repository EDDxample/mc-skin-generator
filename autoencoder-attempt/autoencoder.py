import numpy as np
import PIL
from keras import Sequential, Model
from keras.layers import Input, Conv2D, UpSampling2D, Flatten, Reshape, Dense
from keras.callbacks import Callback
from keras.utils import plot_model


skins = np.load('mc-skins-64x64.npy')


# ========== MODEL ==========


encoder = Sequential()
encoder.add(Dense(64*64, activation='relu', input_shape=(None,64*64*4)))
encoder.add(Dense(32*32, activation='relu'))
encoder.add(Dense(100, activation='relu'))
encoder.compile(optimizer='adam', loss='mse', metrics=['acc'])


decoder = Sequential()
decoder.add(Dense(32*32, activation='relu', input_shape=(None,100)))
decoder.add(Dense(64*64, activation='relu'))
decoder.add(Dense(64*64*4, activation='sigmoid'))
decoder.compile(optimizer='adam', loss='mse', metrics=['acc'])


autoencoder_input = Input(shape=(skins.shape[1],))
x = encoder(autoencoder_input)
autoencoder = Model(autoencoder_input, decoder(x))
autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])

if False:
    plot_model(autoencoder, to_file='model.png', show_shapes=True)
    exit()

#  ========== LOGGER ==========

class Logger(Callback):
    def __init__(self):
        self._epoch = '_'
    def on_epoch_begin(self, epoch, logs): self._epoch = epoch
    def on_batch_end(self, batch, logs):
        if batch % 50 == 0:
            arr = autoencoder.predict(skins[7:8])
            img = arr.reshape(int(arr.shape[1]/256), 64, 4) * 255
            PIL.Image.fromarray(img.astype('uint8')).save(f'output/_{self._epoch}_{batch}.png')
    
cb = Logger()

#  ========== TRAIN ==========

epochs = 20
autoencoder.fit(skins, skins, None, epochs, callbacks=[cb])
autoencoder.save('output/mc-autoencoder.h5')
decoder.save('output/mc-generator.h5')


for i in range(10):
    inp = np.random.normal(loc=0, scale=1, size=[1, decoder.input_shape[1]])
    arr = decoder.predict(inp)

    img = arr.reshape(int(arr.shape[1]/256), 64, 4) * 255
    PIL.Image.fromarray(img.astype('uint8')).save(f'yay{i}.png')