import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import cv2


def build_generator():
    input_layer = Input(shape=(None, None, 1))

    x = Conv2D(64, (3, 3), padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    for _ in range(4):
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    output_layer = Conv2D(1, (3, 3), padding='same', activation='tanh')(x)

    return Model(input_layer, output_layer)


def build_discriminator():
    input_layer = Input(shape=(None, None, 1))

    x = Conv2D(64, (3, 3), strides=2, padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    for _ in range(4):
        x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    output_layer = Dense(1, activation='sigmoid')(x)

    return Model(input_layer, output_layer)


def compile_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    discriminator.trainable = False

    gan_input = Input(shape=(None, None, 1))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)

    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    return gan


def train_gan(generator, discriminator, gan, data, epochs=10000, batch_size=64):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        imgs = data[idx]

        gen_imgs = generator.predict(imgs)

        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = gan.train_on_batch(imgs, real)

        if epoch % 100 == 0:
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")
            save_images(generator, epoch)


def save_images(generator, epoch, examples=3, dim=(1, 3), figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()


if __name__ == "__main__":

    generator = build_generator()
    discriminator = build_discriminator()
    gan = compile_gan(generator, discriminator)


