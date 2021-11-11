import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Masking, BatchNormalization
from tensorflow import squeeze
from DataManager import ImageToVector, num_to_char, characters
from keras.preprocessing import sequence
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
from functools import reduce
from operator import add
import pickle

from Main import csvFile, imagepath, imagename
import time
start_time = time.time()

#DATA
#X, y = ImageToVector(True, data=csvFile, directoryname=imagepath, imagename=imagename)

(x_train_unpadded, y_train_unpadded), (x_valid_unpadded, y_valid_unpadded) = ImageToVector(data=csvFile, directoryname=imagepath, imagename=imagename)
    #train_test_split(X, y,
    #                                                                                          test_size=0.2,
    #                                                                                          random_state=42)

modelSaveLocation = "BrandNewModel/"
batch_size = 32
epochs = 128
early_stopping_patience = 100 # Rausgenommen

labels = y_train_unpadded + y_valid_unpadded
img_height = 32
img_width = None
max_length = max([len(label) for label in labels])
yflat = reduce(add, y_train_unpadded) + reduce(add, y_valid_unpadded)
characters = len(characters)
padding_value = 0.0
padding_y = 0.0 #characters+1

if __name__ == "__main__":
    x_train = sequence.pad_sequences(x_train_unpadded, value=float(padding_value), dtype='float32',
                                     padding="post", truncating='post')
    x_valid = sequence.pad_sequences(x_valid_unpadded, value=float(padding_value), dtype='float32',
                                    padding="post", truncating='post')
    y_train = sequence.pad_sequences(y_train_unpadded, value=float(padding_y),
                                     dtype='float32', padding="post")

    #x_train_len = np.asarray([len(x_train[i]) for i in range(len(x_train))])
    y_train_len = np.asarray([len(y_train[i]) for i in range(len(y_train))])
    y_train_len_max = y_train_len.max()

    y_valid = sequence.pad_sequences(y_valid_unpadded, value=float(padding_y),
                                        dtype='float32', padding="post",maxlen=y_train_len_max)

    def encode_single_sample(img, label):
        return {"image": img, "labels": label} #TODO: label [s/_] ?

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    validation_dataset = (
        validation_dataset.map(
            encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )


#CTC (https://keras.io/examples/vision/captcha_ocr/)
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred): #mask = None
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64") #img width * batch size
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64") #labellength * batch size

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

#MODEL
def build_model():
    # Inputs to the model
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    masked_img = Masking(mask_value=padding_value)(input_img)

    # For calculating loss via CTC-Layer
    labels = layers.Input(name="labels", shape=(y_train_len_max,), dtype="float32")
    masked_labels = Masking(mask_value=padding_y)(labels)

    # 1st conv pool block
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal",
                      strides=1, padding="same", name="Conv1")(masked_img)
    x = layers.MaxPooling2D((2, 2), strides=2, name="pool1")(x)

    # 2nd conv pool block
    x = layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal",
                      strides=1, padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), strides=2, name="pool2")(x)

    # 3rd conv conv pool block
    x = layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal",
                      strides=1, padding="same", name="Conv3")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal",
                      strides=1, padding="same", name="Conv4")(x)
    x = layers.MaxPooling2D((1, 2), strides=(1,2), name="pool3")(x)

    # 4th conv batch block
    x = layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal",
                      strides=1, padding="same", name="Conv5")(x)
    x = BatchNormalization()(x)

    # 5th conv batch pool block
    x = layers.Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal",
                      strides=1, padding="same", name="Conv6")(x)
    x = BatchNormalization()(x)
    x = layers.MaxPooling2D((1, 2), strides=(1,2), name="pool4")(x)

    # 6th conv layer
    x = layers.Conv2D(512, (2, 2), activation="relu", kernel_initializer="he_normal",
                      strides=1, padding="valid", name="Conv7")(x)

    # Map-to-Sequence
    x = squeeze(x, axis=2)
    max_char_count = x.get_shape().as_list()[1]

    # RNNs
    rnn = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    rnn = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25), name="lastLSTM")(rnn) #concatenation (default)

    # Output layer (Copied from tf1.x implementation, where this layer was manually implemented)
    x = layers.Dense(characters+1, activation="softmax", name="dense2")(rnn) # characters + 1

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(masked_labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam() # TODO: find best optimizer; Paper: Adadelta()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model

if __name__ == "__main__":
    # Get the model
    model = build_model()
    model.summary()

    tb_callback = keras.callbacks.TensorBoard(
        log_dir=modelSaveLocation, histogram_freq=0, write_graph=True,
        write_images=False, update_freq='epoch'
    )


    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[tb_callback] #, early_stopping]
    )

        # Model speichern:
    model.save(modelSaveLocation)
    #with open(modelSaveLocation+"/trainHistoryDict", 'wb') as file_pi:
    #    pickle.dump(history.history, file_pi)
        #to load history:
        #history = pickle.load(open('/trainHistoryDict', "rb"))

    # Get the prediction model by extracting layers till the output layer
    # (Leaving out the loss calculating CTC-Layer)
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="lastLSTM").output
    )
    prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][ #TODO: greedy / beamsearch
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def characterfinder(label):
    from DataManager import characters
    l=""
    for i in label:
        if int(i) < len(characters):
            l += characters[int(i)]
    return l

if __name__ == "__main__":
    time = (time.time() - start_time)
    time = time / 60
    if time >= 60:
        print("--- ", time//60, "hours and ", time % 60, "minutes ---")
    else:
        print("--- %s minutes ---" % time)

    #  Let's check results on some validation samples  #Um prediction einheitlich zu machen: characters list muss einheitlich
    for batch in validation_dataset.take(1):           #pro batch take one (first)
        batch_images = batch["image"]
        batch_labels = batch["labels"]
        preds = prediction_model.predict(batch_images) #prediction happens here
        pred_texts = decode_batch_predictions(preds)   #ctc_decode prediction
        orig_texts = []
        j = 0
        for label in batch_labels:
            label = characterfinder(y_valid[j].tolist())
            orig_texts.append(label)
            j += 1

    for i in range(10):
        print(orig_texts[i], " - ", pred_texts[i])


"""for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    j=0
    for i in pred_texts:

        print("prediction:"+ i +"label:"+ characterfinder(y_valid[j].tolist()) )
        j += 1
"""