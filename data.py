import pandas as pd
import tensorflow as tf

IMG_SHAPE = [224, 224, 3]


def read_df(csv_path):
    df = pd.read_csv(csv_path)
    df = df.fillna(0)
    # df[df["Cardiomegaly"] < 0]["Cardiomegaly"] = 0
    img_paths = df["Path"]
    labels = df["Cardiomegaly"].astype(int)
    return img_paths, labels

def read_df_attention(csv_path):
    df = pd.read_csv(csv_path)
    df = df.fillna(0)
    # df[df["Cardiomegaly"] < 0]["Cardiomegaly"] = 0
    img_paths = df["attentionPathFinal"]
    labels = df["Cardiomegaly"].astype(int)
    return img_paths, labels


def crop_center_and_resize(image):
    s = tf.shape(image)
    h, w = s[0], s[1]
    c = tf.minimum(w, h)
    w_start = (w - c) // 2
    h_start = (h - c) // 2
    cropped_image = tf.image.crop_to_bounding_box(image, h_start, w_start, c, c)
    return tf.image.resize(cropped_image, IMG_SHAPE[:2], method=tf.image.ResizeMethod.BILINEAR)

def augment_image(image):
    seq = iaa.Sequential([
        iaa.GaussianBlur((0, 3.0)),  # add bluriness
        iaa.Add((-10, 10), per_channel=0.5), # add brightness
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        # iaa.Rot90((0, 3), keep_size=False),
        sometimes(iaa.CropAndPad(
            percent=(-0.1, 0.1),
            pad_cval=(0, 255)
        )),
        # iaa.AdditiveGaussianNoise(scale=0.05*255),  # add gaussian noise
        iaa.GammaContrast((0.5, 1.8))
    ])
    return seq(images=[image])[0]

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.grayscale_to_rgb(image, name=None)
    image = crop_center_and_resize(image)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image


def build_dataset(csv_path):
    img_paths, labels = read_df(csv_path)
    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    image_ds = path_ds.map(load_and_preprocess_image)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    return dataset

def build_dataset_attention(csv_path):
    img_paths, labels = read_df_attention(csv_path)
    path_ds = tf.data.Dataset.from_tensor_slices(img_paths)
    image_ds = path_ds.map(load_and_preprocess_image)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    return dataset