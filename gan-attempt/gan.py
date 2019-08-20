import numpy as np
from keras import Sequential, Model, Input
from keras.layers import Dense, LeakyReLU, Dropout

def main():
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)
    gan.summary()


def create_generator():
    g = Sequential()
    g.add(Dense(256, input_dim=100)) ; g.add(LeakyReLU(0.2))
    g.add(Dense(512))                ; g.add(LeakyReLU(0.2))
    g.add(Dense(1024))               ; g.add(LeakyReLU(0.2))
    g.add(Dense(2048))               ; g.add(LeakyReLU(0.2))
    g.add(Dense(64*64*4,              activation='sigmoid')) # Tut: tanh
    g.compile(optimizer='adam', loss='mse') # Tut: binary_crossentropy
    return g

def create_discriminator():
    d = Sequential()
    d.add(Dense(2048, input_dim=64*64*4)) ; d.add(LeakyReLU(0.2)) ; d.add(Dropout(0.3))
    d.add(Dense(1024))                    ; d.add(LeakyReLU(0.2)) ; d.add(Dropout(0.3))
    d.add(Dense(512))                     ; d.add(LeakyReLU(0.2)) ; d.add(Dropout(0.3))
    d.add(Dense(256))                     ; d.add(LeakyReLU(0.2))
    d.add(Dense(1,                         activation='sigmoid'))
    d.compile(optimizer='adam', loss='binary_crossentropy')
    return d

def create_gan(discriminator, generator):
    gan_input = Input((100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

if __name__ == "__main__": main()