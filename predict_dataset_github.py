import os
import cv2
import numpy as np
from model import build_model

def load_model(model_path):
    model = build_model()
    model.load_weights(model_path)
    return model

def mask_parse(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask*255

def predict_and_save(model, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

    for image_file in image_files:
        # Construct the full path to the input image
        img_path = os.path.join(input_folder, image_file)

        # Read and preprocess the input image
        img = cv2.imread(img_path)
        #img = cv2.resize(img, (256, 256))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        # Predict using the model
        result = model.predict(img)
        out = mask_parse(result)

        # Construct the output file name without the '.png' extension
        filename_without_extension = os.path.splitext(image_file)[0]

        # Save the result in the output folder as a PNG file
        output_path = os.path.join(output_folder, f"{filename_without_extension}.png")
        cv2.imwrite(output_path, out)

if __name__ == "__main__":
    # Replace 'best_model.h5' with the path to your trained model weights
    model_path = './best_model.h5'
    model = load_model(model_path)

    # Specify input and output folders
    input_folder = 'path to input overlapping cervical cell images'
    output_folder = 'path to output segmentation mask image folder'

    # Run prediction on all images in the input folder and save the results
    predict_and_save(model, input_folder, output_folder)
