import cv2, os
import argparse
import numpy as np
from dotenv import load_dotenv
from keras import models

load_dotenv()

SIZE = 256

print("Please provide absolute paths to the image!")
parser = argparse.ArgumentParser(description="Process an image file.")
parser.add_argument(
    "--path", type=str, help="the path to the image file", required=True
)
args = parser.parse_args()

IMAGE_PATH = args.path


img = cv2.imread(IMAGE_PATH, 1)
img = cv2.resize(img, (SIZE, SIZE))
img = img.astype("float32") / 255.0

model = models.load_model(os.getenv("COLOURIZER_MODEL"))

predicted = np.clip(model.predict(img.reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape(
    SIZE, SIZE, 3
)

predicted_img = (predicted * 255).astype(np.uint8)
image = cv2.cvtColor(predicted_img, cv2.COLOR_YUV2BGR)
cv2.imwrite(f"{os.getenv("RESULTS_PATH")}/output_image.png", image)
