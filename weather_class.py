# *************** SETTINGS *************** #
MODEL_NAME = 'VGG16-TL'
BATCH_SIZE = 6

EPOCHS = 100
EXIF_FLAG = 0 # Set on 1 if the program is running for the first time with a dataset D, 0 otherwise.

MODELS_DIR = 'models/'
trainingset = "Train_New/"

testset1 = "Test_New/"
testset2 = "Weather_Testset/"
blindtest = "BlindTest/"
mytestset = "MyTestSet/"

imgstype = "*/*.jpg"

csv_file = '1743168.csv'

img_w = 136
img_h = 136
# **************************************** #

import tensorflow as tf
import matplotlib.pyplot as plt
import os, glob
from PIL import Image
import csv
import scikitplot as skplt

# Clear info and warnings, showing errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fix for TensorFlow2 + CUDA 10.1 + CUDNN 7.6.5 + Python 3.7.5
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers, applications, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def savemodel(model, problem):
    filename = os.path.join(MODELS_DIR, '%s.h5' % problem)
    model.save(filename)
    print("\nModel saved successfully on file %s\n" % filename)

def loadmodel(problem):
    filename = os.path.join(MODELS_DIR, '%s.h5' % problem)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" % filename)
    except OSError:
        print("\nModel file %s not found!!!\n" % filename)
        model = None
    return model

# batch_size = 64
def RickyNet(input_shape, num_classes):
    model = Sequential()

    model.add(AveragePooling2D(pool_size=(3, 3), strides=(3, 3), input_shape=input_shape))

    model.add(Conv2D(32, kernel_size=(1, 1), activation="relu", padding="valid"))
    model.add(Conv2D(64, kernel_size=(1, 1), activation="relu", padding="valid"))
    model.add(Conv2D(128, kernel_size=(2, 2), activation="relu", padding="valid"))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.45))
    model.add(Dense(512, activation="relu"))

    model.add(Dropout(0.35))
    model.add(Dense(128, activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def delCorruptEXIF(files):
    for f in files:
        if(os.stat(f).st_size):
            print(files.index(f)," ---- ", f)
            image = Image.open(f).convert('RGB')
            data = list(image.getdata())
            image_without_exif = Image.new(image.mode, image.size)
            image_without_exif.putdata(data)
            image_without_exif.save(f)
        else:
            os.remove(f)

def processData(batch_size):
    trd = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.3,
        rotation_range=7,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        validation_split=0.1,
        fill_mode="reflect")

    trg = trd.flow_from_directory(
        directory=trainingset,
        target_size=(img_h, img_w),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        subset='training'
    )

    teg = trd.flow_from_directory(
        directory=trainingset,
        target_size=(img_h, img_w),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        subset='validation'
    )

    return trg, teg

def processTest(batch_size, test_dir):
    ted = ImageDataGenerator(
        rescale=1. / 255)

    teg = ted.flow_from_directory(
        directory=test_dir,
        target_size=(img_h, img_w),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return teg

def plot_history(history, name):
    # Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(name+' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name+' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def printSinglePredictions(model, teg3, classnames):
    x, y = teg3.next()
    for i in range(len(x)):
        pred = classnames[model.predict(x)[i].argmax()]
        print("Class of the image:", classnames[y[i].argmax()], "\t RickyNet prediction:", pred)
        plt.imshow(x[i])
        plt.show()

def savePredictions(model, teg3, classnames):
    i = 0

    while i <= teg3.batch_index and i < 1:
        data= teg3.next()
        pred = model.predict(data)

        for k in range(BATCH_SIZE):
            plt.imshow(data[0][k])
            plt.title(classnames[pred[k].argmax()])
            plt.savefig('PREDICTIONS/img'+str(k)+'.png')
        i+=1

def solveBlind(model, teg3, classnames):
    i = 0
    lines = [None] * teg3.n
    while i <= teg3.batch_index:
        data = teg3.next()
        pred = model.predict(data)

        for k in range(BATCH_SIZE):
            idx = BATCH_SIZE*i + k
            pred_label = classnames[pred[k].argmax()]
            lines[idx] = [pred_label]
            # print(str(int((idx/teg3.n)*100)) + "%")
        i += 1

    with open(csv_file, "w") as fw:
        wr = csv.writer(fw)
        wr.writerows(lines)

def load_backbone_net(input_shape):
    # define input tensor
    input0 = Input(shape=input_shape)

    #model = applications.VGG19(weights="imagenet", include_top=False, input_tensor=input0)
    #model= applications.InceptionV3(weights="imagenet", include_top=False, input_tensor=input0)
    model = applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input0)

    feature_extractor = Model(inputs=input0, outputs=model.output)
    optimizer = 'adam'  # alternative 'SGD'

    feature_extractor.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return feature_extractor
    return model

def transferNet(feature_extractor, num_classes, output_layer_name, trainable_layers):
    # get the original input layer tensor
    input_t = feature_extractor.get_layer(index=0).input

    # set the feature extractor layers as non-trainable
    for idx, layer in enumerate(feature_extractor.layers):
        if layer.name in trainable_layers:
            layer.trainable = True
        else:
            layer.trainable = False

    # get the output tensor from a layer of the feature extractor
    output_extractor = feature_extractor.get_layer(name=output_layer_name).output

    #output_extractor = MaxPooling2D(pool_size=(4,4))(output_extractor)
    output_extractor = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(output_extractor)

    # flat the output of a Conv layer
    flatten = Flatten()(output_extractor)
    flatten_norm = BatchNormalization()(flatten)

    # add a Dense layer
    dense = Dropout(0.4)(flatten_norm)
    dense = Dense(200, activation='relu')(dense)
    dense = BatchNormalization()(dense)

    # add a Dense layer
    dense = Dropout(0.4)(dense)
    dense = Dense(100, activation='relu')(dense)
    dense = BatchNormalization()(dense)

    # add the final output layer
    dense = BatchNormalization()(dense)
    dense = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=input_t, outputs=dense, name="transferNet")

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == "__main__":

    # Removing corrupt EXIF data
    if(EXIF_FLAG):
        delCorruptEXIF(glob.glob(trainingset + imgstype))

    batch_size = BATCH_SIZE
    train_generator, test_generator = processData(batch_size)

    num_samples = train_generator.n
    num_classes = train_generator.num_classes
    input_shape = train_generator.image_shape

    classnames = [k for k, v in train_generator.class_indices.items()]

    print("Image input %s" % str(input_shape))
    print("Classes: %r" % classnames)

    stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=6)

    # ------------------------------- Train the model ----------------------------------- #
    print("Generating the model")
    model = RickyNet(input_shape, num_classes)

    print(model.summary())

    steps_per_epoch = train_generator.n // train_generator.batch_size
    val_steps = test_generator.n // test_generator.batch_size + 1

    try:
        history = model.fit_generator(train_generator, epochs=EPOCHS, verbose=1, callbacks=[stopping],
                                      steps_per_epoch=steps_per_epoch,
                                      validation_data=test_generator,
                                      validation_steps=val_steps)
    except KeyboardInterrupt:
        pass



    savemodel(model, MODEL_NAME)
    
    plot_history(history, MODEL_NAME)
    # ---------------------------------------------------------------------------------- #


    # -------------------------------- Load a model ------------------------------------ #
    # model = loadmodel(MODEL_NAME)
    # ---------------------------------------------------------------------------------- #


    # ------------------------------ Evaluation tests ---------------------------------- #
    print("\nAcc and Loss on Test_New:")
    teg1 = processTest(batch_size, testset1)
    val_steps1 = teg1.n // teg1.batch_size + 1
    # Accuracy + Loss
    loss, acc = model.evaluate_generator(teg1, verbose=1, steps=val_steps1)
    print('Test loss: %f' % loss)
    print('Test accuracy: %f' % acc)

    print("\nAcc and Loss on Weather_Test:")
    teg2 = processTest(batch_size, testset2)
    val_steps2 = teg2.n // teg2.batch_size + 1
    # Accuracy + Loss
    loss, acc = model.evaluate_generator(teg2, verbose=1, steps=val_steps2)
    print('Test loss: %f' % loss)
    print('Test accuracy: %f' % acc)

    # Confusion Matrix
    Y_pred = model.predict_generator(teg2, val_steps2)
    y_pred = np.argmax(Y_pred, axis=1)
    skplt.metrics.plot_confusion_matrix(teg2.classes, y_pred, normalize=True, title="RickyNet")
    plt.ylim([3.5, -.5])
    plt.tight_layout()
    plt.show()

    # Precision Recall Curve
    skplt.metrics.plot_precision_recall_curve(teg2.classes, Y_pred, title="RickyNet")
    plt.show()
    
    # Precision + Recall + f1-score
    preds = model.predict_generator(teg2, verbose=1, steps=val_steps2)
    Ypred = np.argmax(preds, axis=1)
    Ytest = teg2.classes
    print(classification_report(Ytest, Ypred, labels=None, target_names=classnames, digits=3))
    # ---------------------------------------------------------------------------------- #




    # ------------------------------ Transfer learning + Fine tuning ---------------------------------- #
    # load the pre-trained model
    feature_extractor = load_backbone_net(input_shape)
    feature_extractor.summary()

    # VGG16
    name_output_extractor = "block5_pool"
    trainable_layers = ["block5_conv3"]

    # build the transfer model
    transfer_model = transferNet(feature_extractor, num_classes, name_output_extractor, trainable_layers)
    transfer_model.summary()


    steps_per_epoch = train_generator.n // train_generator.batch_size
    val_steps = test_generator.n // test_generator.batch_size + 1

    try:
        history_transfer = transfer_model.fit_generator(train_generator, epochs=EPOCHS, verbose=1, callbacks=[stopping], \
                                                        steps_per_epoch=steps_per_epoch, \
                                                        validation_data=test_generator, \
                                                        validation_steps=val_steps)
    except KeyboardInterrupt:
        pass

    savemodel(transfer_model, MODEL_NAME)
    plot_history(history_transfer, MODEL_NAME)
    # ------------------------------------------------------------------------------------------------- #

    # -------------------------------- Load a TL model ------------------------------------ #
    # transfer_model = loadmodel(MODEL_NAME)
    # ------------------------------------------------------------------------------------- #

    # ------------------------------ TF Evaluation tests ---------------------------------- #
    print("\nAcc and Loss on Test_New:")
    teg1 = processTest(batch_size, testset1)
    val_steps1 = teg1.n // teg1.batch_size + 1
    # Accuracy + Loss
    loss, acc = transfer_model.evaluate_generator(teg1, verbose=1, steps=val_steps1)
    print('Test loss: %f' % loss)
    print('Test accuracy: %f' % acc)

    print("\nAcc and Loss on Weather_Test:")
    teg2 = processTest(batch_size, testset2)
    val_steps2 = teg2.n // teg2.batch_size + 1
    # Accuracy + Loss
    loss, acc = transfer_model.evaluate_generator(teg2, verbose=1, steps=val_steps2)
    print('Test loss: %f' % loss)
    print('Test accuracy: %f' % acc)

    # Precision + Recall + f1-score
    preds = transfer_model.predict_generator(teg2, verbose=1, steps=val_steps2)
    Ypred = np.argmax(preds, axis=1)
    Ytest = teg2.classes
    print(classification_report(Ytest, Ypred[:teg2.n], labels=None, target_names=classnames, digits=3))

    # Confusion Matrix
    Y_pred = transfer_model.predict_generator(teg2, val_steps2)
    y_pred = np.argmax(Y_pred, axis=1)
    skplt.metrics.plot_confusion_matrix(teg2.classes, y_pred[:teg2.n], normalize=True, title=MODEL_NAME)
    plt.ylim([3.5, -.5])
    plt.tight_layout()
    plt.show()

    # Precision Recall Curve
    skplt.metrics.plot_precision_recall_curve(teg2.classes, Y_pred[:teg2.n], title=MODEL_NAME)
    plt.show()
    # ------------------------------------------------------------------------------------ #

    # ---------------------------------- Blind test -------------------------------------- #
    tegb = processTest(batch_size, blindtest)
    solveBlind(transfer_model, tegb, classnames)
    # ------------------------------------------------------------------------------------ #

