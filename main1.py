from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import random
from dataloader1 import Dataloader
from preparedata import Preparedata
from model6 import build_model, EditDistanceCallback, decode_batch_predictions
np.random.seed(42)
tf.random.set_seed(42)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def preprocess_image(image_path, image_width, image_height):
    img_size=(image_width, image_height)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def distortion_free_resize(image, img_size):
        w, h = img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        # Check tha amount of padding needed to be done.
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )

        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        return image


def main():

    dl = Dataloader("Line_Data_10_Labels.txt")
    words_list = dl.loader()
    random.shuffle(words_list)
    #train_samples, validation_samples, test_samples = dl.datasplit(words_list)
    train_samples, validation_samples = dl.datasplit(words_list)

    print(f"Total training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(validation_samples)}")
    #print(f"Total test samples: {len(test_samples)}")

    train_img_paths, train_labels = dl.get_image_paths_and_labels(train_samples)
    validation_img_paths, validation_labels = dl.get_image_paths_and_labels(validation_samples)
    #test_img_paths, test_labels = dl.get_image_paths_and_labels(test_samples)

    #print(f"Sample of train_img_paths", train_img_paths[:10])
    #print(f"Sample of train_labels", train_labels[:10])

    train_labels_cleaned, max_len, characters = dl.train_clean_labels(train_labels)
    print("Size of character set:", len(characters))
    print("Maximum length:", max_len)
    validation_labels_cleaned = dl.clean_labels(validation_labels)
    #test_labels_cleaned = dl.clean_labels(test_labels)

    print(characters)
    # save characters and words
    f_char = open('Characters_10.txt', 'w', encoding='utf-8')
    for ele in characters:
        f_char.write(ele)
    f_char.close()
    # Building the character vocabulary
    # Keras provides different preprocessing layers to deal with different modalities of data. 
    # This guide provids a comprehensive introduction. Our example involves preprocessing labels 
    # at the character level. This means that if there are two labels, e.g. "cat" and "dog", then 
    # our character vocabulary should be {a, c, d, g, o, t} (without any special tokens). We use 
    # the StringLookup layer for this purpose.
    AUTOTUNE = tf.data.AUTOTUNE

    # Mapping characters to integers.
    char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters.
    num_to_char = StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

    
    batch_size = 75
    padding_token = 199
    image_width = 800
    image_height = 64

    pd = Preparedata(batch_size, padding_token, image_width, image_height, char_to_num, max_len)
    train_ds = pd.prepare_dataset(train_img_paths, train_labels_cleaned, AUTOTUNE)
    validation_ds = pd.prepare_dataset(validation_img_paths, validation_labels_cleaned, AUTOTUNE)
    #test_ds = pd.prepare_dataset(test_img_paths, test_labels_cleaned, AUTOTUNE)
    
    #print("Test DS shape:",test_ds.shape)
    # Get the model.
    model = build_model(image_width, image_height, char_to_num)
    model.summary()

    validation_images = []
    validation_labels = []

    for batch in validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])


    #model = build_model()
    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    edit_distance_callback = EditDistanceCallback(prediction_model, validation_images, validation_labels, max_len)

    epochs = 3  # To get good results this should be at least 50.

    checkpoint_path = "saved_model_line_10\hindi_model"
    csv_logger = keras.callbacks.CSVLogger('log.csv', append=True, separator=';');
    for i in range(1,15):
        if i!=1:
            model = keras.models.load_model(checkpoint_path+str(epochno)+'.tf')
        epochno = i*epochs
        # Train the model.
        history = model.fit(train_ds,
            validation_data=validation_ds,
            epochs=epochs,
            callbacks=[edit_distance_callback, csv_logger],)
        model.save(checkpoint_path+str(epochno)+'.tf')
        prediction_model.save('saved_model_line_10\pred_model')
        prediction_model.save('saved_model_line_10\pred_model_Arch4.h5')

    # Load the previously saved weights
    #prediction_model1 = keras.models.load_model('saved_model\pred_model')
    #predFile = open("predFile.txt", "a", encoding="utf-8")
    test_img = preprocess_image("LineData/Line_316.png", image_width, image_height)
    print(test_img.shape)
    test_img = tf.expand_dims(test_img, axis=0)
    #print(test_img.shape)
    
    pred = prediction_model.predict(test_img)
    pred_text = decode_batch_predictions(pred, max_len, num_to_char)
    predFile.write(str(pred_text)+"\n")
    print(pred_text)
    print("Complete run")
if __name__ == '__main__':
    main()