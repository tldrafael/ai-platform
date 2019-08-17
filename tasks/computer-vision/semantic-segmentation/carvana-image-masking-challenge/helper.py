import numpy as np
import cv2
import random
import re
import imageio

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
import keras.backend as K


def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)
    return Model(inputs=inputs, outputs=classify)


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def preprocess_input(img, input_shape=(128, 128)):
    return cv2.resize(img, input_shape) / 255.0


def image_generator(filepaths, setname=None, batch_size=2, input_shape=(128, 128),
                    output_shape=(128, 128), aug=False):
    if aug:
        aug_gen = ImageDataGenerator(shear_range=0.25, rotation_range=30,
                                     width_shift_range=0.25,
                                     height_shift_range=0.25, zoom_range=0.75,
                                     horizontal_flip=True, vertical_flip=True)
    while True:
        trainfiles = filepaths.copy()
        if setname == 'train':
            trainfiles = trainfiles[:int(len(trainfiles) * 0.8)]
        elif setname == 'val':
            trainfiles = trainfiles[int(len(trainfiles) * 0.8):]

        random.shuffle(trainfiles)
        trainfiles_masks = [re.sub('/train', '/train_masks', f) for f in trainfiles]
        trainfiles_masks = [re.sub('\.jpg$', '_mask.gif', f) for f in trainfiles_masks]

        for i in range(0, len(trainfiles), batch_size):
            if (i + 1) * batch_size > len(trainfiles):
                batch_files = trainfiles[(len(trainfiles) - batch_size):]
                batch_filesmask = trainfiles_masks[(len(trainfiles) - batch_size):]
            else:
                batch_files = trainfiles[(i * batch_size):((i + 1) * batch_size)]
                batch_filesmask = trainfiles_masks[(i * batch_size):((i + 1) * batch_size)]

            batch_imgs = np.zeros((batch_size, input_shape[0], input_shape[1], 3))
            batch_masks = np.zeros((batch_size, output_shape[0], output_shape[1], 1))
            for j in range(batch_size):
                batch_imgs[j] = preprocess_input(cv2.imread(batch_files[j]), input_shape)
                tmp_mask = cv2.resize(imageio.mimread(batch_filesmask[j])[0], output_shape)
                batch_masks[j] = np.expand_dims(tmp_mask, axis=2) // 255

            if aug:
                for j in range(batch_size):
                    seed = np.random.randint(1e6)
                    batch_imgs[j] = aug_gen.random_transform(batch_imgs[j], seed=seed)
                    batch_masks[j] = aug_gen.random_transform(batch_masks[j], seed=seed)

            yield (batch_imgs, batch_masks)


def rle_encode(img, output_shape):
    img = np.reshape(img, output_shape)
    img = cv2.resize(img, (1280, 1918))
    img = np.where(img > 0.5, 1, 0).astype(np.uint8)
    pixels = img.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)


def show_arr(arr):
    cv2.imshow('', arr)
    cv2.waitKey()
    cv2.destroyAllWindows()
