
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from PIL import Image
import torch
import torchvision.transforms as transforms


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            ind = 0
            while True:
                ret_val, frame = video_capture.read()
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
                if keyCode == ord('p'):
                    cv2.imwrite('image{}.jpg'.format(ind) )
                    ind += 1
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")





if __name__ == "__main__":

    base = 'torch'
    base = 'tf'

    if base == 'torch':
        model = torch.load('Resnet_torch_model.pt', map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Set the model to evaluation mode
        model.eval()

        # Define transformation for input images
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((112, 112)),  # Decrease resolution to 112x112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Function to preprocess frames and predict using the model
        def predict(frame):
            image = transform(frame).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                outputs = model(image)
            # Process outputs here...
            return outputs
        
    elif base == 'tf':
        model = load_model('vgg16_transferlearning_tf.h5')

        # Define transformation for input images
        def preprocess_image(frame):
            img = Image.fromarray(frame)
            img = img.resize((224, 224))  # Resize to match VGG16 input size
            img = np.array(img)
            img = preprocess_input(img)
            return img

        # Function to preprocess frames and predict using the model
        def predict(frame):
            image = preprocess_image(frame)
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            outputs = model.predict(image)
            # Process outputs here...
            return outputs


    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            # ind = 0
            frames_to_skip = 2  # Adjust as needed
            frame_count = 0


            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    break

                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break 

                # Stop the program on the ESC key or 'q'
                keyCode = cv2.waitKey(2) & 0xFF #10
                if keyCode == 27 or keyCode == ord('q'):
                    break
                
                frame_count += 1
                if frame_count % frames_to_skip != 0:
                    continue  # Skip frames
                
                # Perform prediction
                outputs = predict(frame)
                
                # Display the frame or process further as needed
                cv2.imshow('Frame', frame)


        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")
