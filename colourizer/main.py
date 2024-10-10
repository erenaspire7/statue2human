import cv2, os, re
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from keras.api import layers, models, Model, optimizers
from keras.api.preprocessing.image import img_to_array


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


def colourizer_model():
    inputs = layers.Input(shape=[SIZE, SIZE, 3])

    d1 = down(128, (3, 3), False)(inputs)
    d2 = down(128, (3, 3), False)(d1)
    d3 = down(256, (3, 3), True)(d2)
    d4 = down(512, (3, 3), True)(d3)

    d5 = down(512, (3, 3), True)(d4)

    u1 = up(512, (3, 3), False)(d5)
    u1 = layers.concatenate([u1, d4])

    u2 = up(256, (3, 3), False)(u1)
    u2 = layers.concatenate([u2, d3])

    u3 = up(128, (3, 3), False)(u2)
    u3 = layers.concatenate([u3, d2])

    u4 = up(128, (3, 3), False)(u3)
    u4 = layers.concatenate([u4, d1])

    u5 = up(3, (3, 3), False)(u4)
    u5 = layers.concatenate([u5, inputs])

    output = layers.Conv2D(3, (2, 2), strides=1, padding="same")(u5)
    return Model(inputs=inputs, outputs=output)


statue_dataset = load_dataset(os.getenv("STATUE_PATH"), False)
human_dataset = load_dataset(os.getenv("HUMAN_PATH"))

statue_tensor = np.reshape(human_dataset, (len(human_dataset), SIZE, SIZE, 3))
human_tensor = np.reshape(human_dataset, (len(human_dataset), SIZE, SIZE, 3))


model = colourizer_model()
model.summary()


model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="mean_absolute_error",
    metrics=["acc"],
)

model.fit(statue_tensor, human_tensor, epochs=50, batch_size=50, verbose=1)
model.save("colourizer.keras")
