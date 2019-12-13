import tensorflow as tf
from data import IMG_SHAPE


def get_model():
    res_net = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    res_net.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    middle_layer_1 = tf.keras.layers.Dense(64, activation="relu")
    bn_1 = tf.keras.layers.BatchNormalization()
    middle_layer_2 = tf.keras.layers.Dense(16, activation="relu")
    bn_2 = tf.keras.layers.BatchNormalization()
    prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential([
        res_net,
        global_average_layer,
        middle_layer_1,
        bn_1,
        middle_layer_2,
        bn_2,
        prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalsePositives()])

    print(model.summary())
    return model

def get_global_model():
    res_net = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    res_net.trainable = True

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    model = tf.keras.Sequential([
        res_net,
        global_average_layer,
        prediction_layer
    ])

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.FalsePositives()])

    print(model.summary())
    return model
