###### SPATIAL ATTENTION ######
import os

import cv2
import numpy as np
from keras import Model
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten, Concatenate, Lambda
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add, ReLU

import tensorflow as tf
from keras.activations import sigmoid
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow_core.python.keras.applications.vgg19 import VGG19
from tqdm import tqdm
from keras.models import load_model
from numpy.random import randint


def down(x, channels):
    x0, x1, x2, x3 = x.shape
    dwconv_3 = SeparableConv2D(filters=channels, kernel_size=3, dilation_rate=2, strides=2, padding="same")(x)
    dwconv_3 = BatchNormalization()(dwconv_3)

    dwconv_7 = SeparableConv2D(filters=channels, kernel_size=3, dilation_rate=4, strides=2, padding="same")(x)
    dwconv_7 = BatchNormalization()(dwconv_7)

    paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])

    if dwconv_3.shape[1] != int(x1 / 2) or dwconv_7.shape[1] != int(x1 / 2):
        if dwconv_3.shape[1] > dwconv_7.shape[1]:
            dwconv_3 = dwconv_3[:, 0:-1, 0:-1, :]
            dwconv_7 = tf.pad(dwconv_7, paddings, "REFLECT")
        elif dwconv_7.shape[1] > dwconv_3.shape[1]:
            dwconv_3 = tf.pad(dwconv_3, paddings, "REFLECT")
            dwconv_7 = dwconv_7[:, 0:-1, 0:-1, :]

    if dwconv_3.shape[1] == int(x1 / 2) and dwconv_7.shape[1] == int(x1 / 2):
        dwconv_3 = tf.pad(dwconv_3, paddings, "REFLECT")
        dwconv_7 = tf.pad(dwconv_7, paddings, "REFLECT")

    f = Concatenate()([dwconv_3, dwconv_7])
    return f




def gate(x, channels, reduction_ratio=16):
    g = Dense(channels // reduction_ratio, use_bias=False)(x)
    g = BatchNormalization()(g)
    g = Activation('relu')(g)
    g = Dense(channels, use_bias=False)(g)
    return g

###### CHANNEL ATTENTION ######

def spatial(x, dilation=1):
    x1, x2, x3 = x.shape[1], x.shape[2], x.shape[1]
    x = down(x, x3)
    x = SeparableConv2D(filters=x3 * 2, kernel_size=2, dilation_rate=dilation)(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    return x

def channel(x, reduction_ratio=16):
    x1, x2, x3 = x.shape[1], x.shape[3], x.shape[2]
    x = tf.reshape(Flatten()(x), [-1, x1, x2, x3])
    x_avg = tf.reshape(Flatten()(GlobalAvgPool2D()(x)), [-1, 1, x1, 1])
    x_max = tf.reshape(Flatten()(GlobalMaxPool2D()(x)), [-1, 1, x1, 1])
    x = tf.keras.layers.concatenate([x_avg, x_max], axis=1)
    x = Conv2D(1, kernel_size=(2, 1))(x)
    x = Flatten()(x)
    x = gate(x, x1, reduction_ratio)
    x = tf.reshape(Flatten()(x), [-1, x1, 1, 1])
    return x

###### COMBINED ATTENTION BLOCK ######
def attn_mech(x, f):
    att_c = channel(x, reduction_ratio=16)
    att_s = spatial(x)
    w = sigmoid(tf.multiply(att_c, att_s))
    w = Conv2D(f, (1, 1))(w)
    return w

#########################################################################

# Define blocks to build the generator
def res_block(ip):
    res_model = Conv2D(64, (3, 3), padding="same")(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1, 2])(res_model)

    res_model = Conv2D(64, (3, 3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)

    return add([ip, res_model])


def upscale_block(ip):
    up_model = Conv2D(256, (3, 3), padding="same")(ip)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1, 2])(up_model)

    return up_model


# Generator model
def create_gen(gen_ip, num_res_block):
    layers = Conv2D(64, (9, 9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1, 2])(layers)

    temp = layers

    for i in range(num_res_block):
        layers = res_block(layers)

    layers = Conv2D(64, (3, 3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)

    attn = attn_mech(temp, 64)
    layers = add([layers, attn])

    # layers = add([layers, temp])

    layers = upscale_block(layers)
    layers = upscale_block(layers)

    op = Conv2D(3, (9, 9), padding="same")(layers)

    return Model(inputs=gen_ip, outputs=op)


# Descriminator block that will be used to construct the discriminator
def discriminator_block(ip, filters, strides=1, bn=True):
    disc_model = Conv2D(filters, (3, 3), strides=strides, padding="same")(ip)

    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)

    disc_model = LeakyReLU(alpha=0.2)(disc_model)

    return disc_model


# Descriminartor, as described in the original paper
def create_disc(disc_ip):
    df = 64

    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df * 2)
    d4 = discriminator_block(d3, df * 2, strides=2)
    d5 = discriminator_block(d4, df * 4)
    d6 = discriminator_block(d5, df * 4, strides=2)
    d7 = discriminator_block(d6, df * 8)
    d8 = discriminator_block(d7, df * 8, strides=2)

    d8_5 = Flatten()(d8)
    d9 = Dense(df * 16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)

    return Model(disc_ip, validity)




def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)

    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)


# Combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)

    gen_features = vgg(gen_img)

    disc_model.trainable = False
    validity = disc_model(gen_img)

    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])


lr_list = os.listdir("lr_images")
hr_list = os.listdir("hr_images")

lr_images = []
print("SAVING NPZ")
for img in lr_list:
    img_lr = cv2.imread("lr_images/" + img)
    # img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2GRAY)
    # img_lr = np.reshape(img_lr, (64, 64, 1))
    lr_images.append([img_lr])


hr_images = []
for img in hr_list:
    img_hr = cv2.imread("hr_images/" + img)
    # img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2GRAY)
    # img_hr = np.reshape(img_hr, (128, 128, 1))
    hr_images.append([img_hr])

lr_images = np.array(lr_images)
hr_images = np.array(hr_images)
np.savez("lr_images", lr_images=lr_images, hr_images=hr_images)

print("LOADING NPZ")
dd = np.load("lr_images.npz")
lr_images = dd["lr_images"]
hr_images = dd["hr_images"]


# Scale values
lr_images = lr_images / 255.
hr_images = hr_images / 255.

# Split to train and test
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images,
                                                         test_size=0.33, random_state=42)

lr_train = np.reshape(lr_train, (lr_train.shape[0], 64, 64, 3))
hr_train = np.reshape(hr_train, (hr_train.shape[0], 256, 256, 3))
lr_test = np.reshape(lr_test, (lr_test.shape[0], 64, 64, 3))
hr_test = np.reshape(hr_test, (hr_test.shape[0], 256, 256, 3))

print(lr_train.shape)
print(hr_train.shape)

hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])
lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

generator = create_gen(lr_ip, num_res_block=16)
generator.summary()

discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
# discriminator.summary()

vgg = build_vgg((256, 256, 3))
# print(vgg.summary())
vgg.trainable = False

gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)

# 2 losses... adversarial loss and content (VGG) loss
# AdversariaL: is defined based on the probabilities of the discriminator over all training samples
# use binary_crossentropy

# Content: feature map obtained by the j-th convolution (after activation)
# before the i-th maxpooling layer within the VGG19 network.
# MSE between the feature representations of a reconstructed image
# and the reference image.
gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")
# gan_model.summary()

# Create a list of images for LR and HR in batches from which a batch of images
# would be fetched during training.
batch_size = 1
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])

epochs = 10
# Enumerate training over epochs
for e in range(epochs):

    fake_label = np.zeros((batch_size, 1))  # Assign a label of 0 to all fake (generated images)
    real_label = np.ones((batch_size, 1))  # Assign a label of 1 to all real images.

    # Create empty lists to populate gen and disc losses.
    g_losses = []
    d_losses = []

    # Enumerate training over batches.
    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b]  # Fetch a batch of LR images for training
        hr_imgs = train_hr_batches[b]  # Fetch a batch of HR images for training

        fake_imgs = generator.predict_on_batch(lr_imgs)  # Fake images

        # First, train the discriminator on fake and real HR images.
        discriminator.trainable = True
        d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)

        # Now, train the generator by fixing discriminator as non-trainable
        discriminator.trainable = False

        # Average the discriminator loss, just for reporting purposes.
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

        # Extract VGG features, to be used towards calculating loss
        image_features = vgg.predict(hr_imgs)

        # Train the generator via GAN.
        # Remember that we have 2 losses, adversarial loss and content (VGG) loss
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])

        # Save losses to a list so we can average and report.
        d_losses.append(d_loss)
        g_losses.append(g_loss)

    # Convert the list of losses to an array to make it easy to average
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)

    # Calculate the average losses for generator and discriminator
    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)

    # Report the progress during training.
    print("epoch:", e + 1, "g_loss:", g_loss, "d_loss:", d_loss)

    # if (e + 1) % 10 == 0:  # Change the frequency for model saving, if needed
    #     # Save the generator after every n epochs (Usually 10 epochs)
    generator.save("models4/gen_e_" + str(e + 1) + ".h5")

###################################################################################
# Test - perform super resolution using saved generator model

generator = load_model('models4/gen_e_10.h5', compile=False)

[X1, X2] = [lr_test, hr_test]
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# generate image from source
gen_image = generator.predict(src_image)

# plot all three images

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('LR Image')
plt.imshow(src_image[0, :, :, :])
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(gen_image[0, :, :, :])
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(tar_image[0, :, :, :])

plt.show()




#### Only SRGAN
#### epoch: 10 g_loss: 5.434879090247699 d_loss: [0.04499957 0.99187875]