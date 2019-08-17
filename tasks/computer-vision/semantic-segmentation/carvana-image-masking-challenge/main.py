import os
import sys
import numpy as np
import glob
import keras
import helper
import mlflow
import mlflow.keras
import cv2


def train(optimizer, epochs, batch_size):
    carvana_trainset = os.getenv('CARVANA_DATASET')
    trainfiles = list(glob.iglob('{}/train/*'.format(carvana_trainset)))

    model = helper.get_unet_128()
    model.summary()
    model.compile(optimizer=optimizer, loss=helper.bce_dice_loss, metrics=[helper.dice_coeff])

    hist = model.fit_generator(helper.image_generator(trainfiles, setname='train', batch_size=batch_size, aug=True),
                               steps_per_epoch=np.ceil(int(len(trainfiles) * 0.8) / batch_size),
                               epochs=epochs,
                               validation_data=helper.image_generator(trainfiles, setname='val', batch_size=batch_size,
                                                                      aug=False),
                               validation_steps=np.ceil(int(len(trainfiles) * 0.2) / batch_size))

    with mlflow.start_run():
        mlflow.log_param('optimizer', optimizer)
        mlflow.log_param('epochs', epochs)
        mlflow.log_metric("loss", hist.history['loss'][-1])
        mlflow.log_metric("val_loss", hist.history['val_loss'][-1])
        mlflow.log_metric("dice_coeff", hist.history['dice_coeff'][-1])
        mlflow.log_metric("val_dice_coeff", hist.history['val_dice_coeff'][-1])
        mlflow.keras.log_model(model, "model")


def predict(uri, imgpath):
    keras.losses.bce_dice_loss = helper.bce_dice_loss
    keras.metrics.dice_coeff = helper.dice_coeff
    model = mlflow.keras.load_model("runs:/{}/model".format(uri))

    img_arr = cv2.imread(imgpath)
    img_arr = helper.preprocess_input(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)

    img_pred = model.predict(img_arr)[0]
    img_pred = np.where(img_pred > 0.5, 255, 0).astype(np.uint8)

    img_pred = cv2.resize(np.repeat(img_pred, 3, axis=2), (256, 256))
    img_arr = cv2.resize(cv2.imread(imgpath), (256, 256))
    helper.show_arr(np.concatenate([img_arr, img_pred], axis=1))


if __name__ == '__main__':
    entrypoint = sys.argv[1]
    if entrypoint == 'train':
        optimizer = sys.argv[2]
        epochs = int(sys.argv[3])
        batch_size = int(sys.argv[4])
        train(optimizer, epochs, batch_size)
    elif entrypoint == 'predict':
        uri = sys.argv[2]
        imgpath = sys.argv[3]
        predict(uri, imgpath)
    else:
        print('Choose an entrypoint: train or test')
