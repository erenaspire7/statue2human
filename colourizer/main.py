import cv2, os, re
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from keras.api import layers, models, Model, optimizers
from keras.api.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

load_dotenv()

# requires a large amount of RAM,
# reduce resolution if system requirements are not enough
SIZE = 256


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def load_dataset(path, yuv=True):
    arr = []

    files = os.listdir(path)
    files = sorted_alphanumeric(files)

    for i in tqdm(files):
        img = cv2.imread(path + "/" + i, 1)

        if yuv:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype("float32") / 255.0
        arr.append(img_to_array(img))

    return arr


def down(filters, kernel_size, apply_batch_normalization=True):
    downsample = models.Sequential()
    downsample.add(layers.Conv2D(filters, kernel_size, padding="same", strides=2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add(layers.LeakyReLU())
    return downsample


def up(filters, kernel_size, dropout=True):
    upsample = models.Sequential()
    upsample.add(
        layers.Conv2DTranspose(filters, kernel_size, padding="same", strides=2)
    )
    if dropout:
        upsample.add(layers.Dropout(0.2))
    upsample.add(layers.LeakyReLU())
    return upsample


def base_model(filter_base):
    inputs = layers.Input(shape=[SIZE, SIZE, 3])

    d1 = down(filter_base, (3, 3), False)(inputs)
    d2 = down(filter_base, (3, 3), False)(d1)
    d3 = down(filter_base * 2, (3, 3), True)(d2)
    d4 = down(filter_base * 4, (3, 3), True)(d3)

    d5 = down(filter_base * 4, (3, 3), True)(d4)

    # upsampling
    u1 = up(filter_base * 4, (3, 3), False)(d5)
    u1 = layers.concatenate([u1, d4])

    u2 = up(filter_base * 2, (3, 3), False)(u1)
    u2 = layers.concatenate([u2, d3])

    u3 = up(filter_base, (3, 3), False)(u2)
    u3 = layers.concatenate([u3, d2])

    u4 = up(filter_base, (3, 3), False)(u3)
    u4 = layers.concatenate([u4, d1])

    u5 = up(3, (3, 3), False)(u4)
    u5 = layers.concatenate([u5, inputs])

    output = layers.Conv2D(3, (2, 2), strides=1, padding="same")(u5)
    return tf.keras.Model(inputs=inputs, outputs=output)


statue_dataset = load_dataset(os.getenv("STATUE_PATH"), False)
human_dataset = load_dataset(os.getenv("HUMAN_PATH"))

statue_tensor = np.reshape(statue_dataset, (len(statue_dataset), SIZE, SIZE, 3))
human_tensor = np.reshape(human_dataset, (len(human_dataset), SIZE, SIZE, 3))

statue_train, statue_test = train_test_split(statue_tensor, test_size=0.2, random_state=42)
human_train, human_test = train_test_split(human_tensor, test_size=0.2, random_state=42)

BASE_FEATURES = 128
LOW_RES = 64
HIGH_RES = 256

# replace constant with above to tweak feature resolution
model = base_model(BASE_FEATURES)
model.summary()


model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="mean_absolute_error",
    metrics=["acc"],
)

batch_size = int(os.getenv("BATCH_SIZE"))

model.fit(statue_train, human_tensor, epochs=50, batch_size=batch_size, verbose=1)
model.evaluate(statue_test,human_test)
model.save("colourizer.keras")
