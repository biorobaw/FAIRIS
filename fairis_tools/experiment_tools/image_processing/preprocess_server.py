import socket
import cv2
import numpy as np
import os
import time

# Ensure working directory is correct
os.chdir("../../..")
print(os.getcwd())


# Feature extraction functions
def extract_hog_features(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor(_winSize=(64, 128),
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    # Resize the image to the window size expected by HOG descriptor
    resized_image = cv2.resize(gray_image, (64, 128))
    hog_features = hog.compute(resized_image).flatten()
    return hog_features


def extract_color_histogram(image, bins=(8, 8, 8)):
    # Calculate histogram in RGB color space
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def extract_spatial_histogram(image, size=(32, 32)):
    # Resize and flatten the image
    downscaled_image = cv2.resize(image, size).flatten()
    return downscaled_image


def extract_combined_features(image):
    hog_features = extract_hog_features(image)
    color_histogram = extract_color_histogram(image)
    spatial_histogram = extract_spatial_histogram(image)

    # Concatenate all features into a single feature vector
    combined_features = np.concatenate([hog_features, color_histogram, spatial_histogram])
    return combined_features


# Socket setup
HOST = 'localhost'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_server:
    socket_server.bind((HOST, PORT))
    socket_server.listen()
    print("Server listening on port", PORT)

    while True:
        # Accept connection from Webots client
        conn, addr = socket_server.accept()
        print('Connected by', addr)

        with conn:
            while True:
                # Wait for Webots signal
                data = conn.recv(1024)
                if not data:
                    break  # Exit the inner loop if no data is received (client closed connection)

                if data == b'Image ready':
                    # Load and process the image
                    image = cv2.imread("data/DataCache/pov.png")
                    if image is not None:
                        features = extract_combined_features(image)
                        features_data = features.astype(np.float32).tobytes()

                        # Send the processed features back to Webots
                        conn.sendall(features_data)
                        print("Sent processed features")
                    else:
                        print("Error: Image file not found or could not be loaded.")

                # Brief pause to avoid busy waiting
                time.sleep(0.1)
