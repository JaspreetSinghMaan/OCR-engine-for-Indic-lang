import tensorflow as tf

class Preparedata:

    def __init__(self, batch_size, padding_token, image_width, image_height, char_to_num, max_len):
        self.batch_size = batch_size
        self.padding_token = padding_token
        self.image_width = image_width
        self.image_height = image_height
        self.char_to_num = char_to_num
        self.max_len = max_len
    
    def preprocess_image(self, image_path):
        img_size=(self.image_width, self.image_height)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = self.distortion_free_resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image


    def vectorize_label(self, label):
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=self.padding_token)
        return label


    def process_images_labels(self, image_path, label):
        image = self.preprocess_image(image_path)
        label = self.vectorize_label(label)
        return {"image": image, "label": label}


    def prepare_dataset(self, image_paths, labels, AUTOTUNE):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
            self.process_images_labels, num_parallel_calls=AUTOTUNE
        )
        return dataset.batch(self.batch_size).cache().prefetch(AUTOTUNE)

    def distortion_free_resize(self, image, img_size):
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

