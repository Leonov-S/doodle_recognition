import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

import os
import urllib.request

os.system('cls' if os.name == 'nt' else 'clear')
class_names = []
## data loading

def download_datasets():
    if (not os.path.exists("data")):
        os.mkdir("data")
    f= open("mini_classes.txt","r")
    classes = f.readlines()
    f.close()
    classes = [c.replace('\n','').replace(' ','_') for c in classes]
    ind = 0
    for c in classes:
        c_url = c.replace('_', '%20')
        print("Downloading " + c + " - [" + str(ind) + " /" + str(len(classes)) + "]")
        full_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/" + c_url + ".npy"
        if (not os.path.exists('data/' + c + ".npy")):
            urllib.request.urlretrieve(full_url, 'data/' + c + ".npy")
        ind += 1
    print("Downloaded " + "[" + str(ind) + " /" + str(len(classes)) + "]")
    

def process_data(vfold_ratio=0.2, max_items_per_class=4000):
    all_files = ['data/' + s for s in os.listdir("data")]
    x_data = np.empty([0, 784])
    y_labels = np.empty([0])

    for idx, file in enumerate(all_files):
        data = np.load(file)
        data = data[0:max_items_per_class, :]
        labels = np.full(data.shape[0], idx)

        x_data = np.concatenate((x_data, data), axis=0)
        y_labels = np.append(y_labels, labels)

        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)
    
    data = None
    labels = None

    permutation = np.random.permutation(y_labels.shape[0])

    x_data = x_data[permutation, :]
    y_labels = y_labels[permutation]

    vfold_size = int(x_data.shape[0]/100*(vfold_ratio*100))

    data_test = x_data[0:vfold_size, :]
    labels_test = y_labels[0:vfold_size]

    data_train = x_data[vfold_size:x_data.shape[0], :]
    labels_train = y_labels[vfold_size:x_data.shape[0]]

    return data_train, labels_train, data_test, labels_test, class_names

def doodle_recognition():
    x_train, y_train, x_test, y_test, class_names = process_data()
    num_classes = len(os.listdir("data"))
    image_size = 28

    plt.figure(figsize=(10, 10))

    # for i in range(54):
    #     ax = plt.subplot(9, 6, i + 1)
    #     plt.imshow(x_train[i].reshape(28, 28))
    #     plt.title(int(y_train[i]))
    #     plt.axis("off")
    # plt.show()
    ## pre-processing

    x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
    print("-----", x_train.shape[0])
    x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Define model
    model = keras.Sequential()
    model.add(layers.Convolution2D(16, (3, 3),
                            padding='same',
                            input_shape=x_train.shape[1:], activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'))
    model.add(layers.MaxPooling2D(pool_size =(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(os.listdir("data")), activation='softmax')) 

    ## Train model
    adam = tf.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                optimizer=adam,
                metrics=['top_k_categorical_accuracy'])
    # print(model.summary())

    model.fit(x = x_train, y = y_train, validation_split=0.1, batch_size = 256, verbose=2, epochs=3)
    model.save('dr.h5')
    model.save_weights('drWeight.h5')
    return model

# ## try model
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

def recognize_img(im, model):
    pred = model.predict(np.expand_dims(im, axis=0))[0]
    ind = (-pred).argsort()
    c_names = [s.replace('.npy', '') for s in os.listdir("data")]

    latex = [c_names[x] for x in ind]
    return latex

if __name__ == "__main__":
    download_datasets()
    doodle_recognition()
