import streamlit as st
from PIL import ImageGrab
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

# Function to release the camera and close OpenCV windows
def release_camera():
    cap.release()
    cv2.destroyAllWindows()

# Function to remove the background from an image and convert it to black and white
def remove_background(image):
    lower_bound = np.array([0, 0, 0], dtype="uint8")
    upper_bound = np.array([50, 50, 50], dtype="uint8")
    mask = cv2.inRange(image, lower_bound, upper_bound)
    image_no_bg = cv2.bitwise_and(image, image, mask=mask)
    image_bw = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image_bw, 1, 255, cv2.THRESH_BINARY)
    return thresholded

# Load the model
model_dict = pickle.load(open('./new_rago_model.p', 'rb'))
model = model_dict['model']

# Initialize the webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Increase min_detection_confidence for better hand landmark detection
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

labels_dict = {
    0: "./images/1.jpg", 1: './images/2.jpg', 2: './images/3.jpg', 3: './images/4.jpg', 4: './images/5.jpg', 5: './images/6.jpg', 6: './images/7.jpg',
    7: './images/8.jpg', 8: './images/9.jpg', 9: './images/10.jpg', 10: './images/11.jpg', 11: './images/12.jpg', 12: './images/0.jpg'
}
st.markdown(
    "<div style='background-color: #f2f2f2; padding: 20px; text-align: center; border-radius: 10px;'><h1 style='color: black;'>BSL - Rago Chu-Ngyi</h1></div>",
    unsafe_allow_html=True,
)

# Define the CSS style for the two columns
column_style = "background-color: white; border-radius: 10px; box-shadow: 0px 0px 10px 2px rgba(0, 0, 0, 0.1); padding: 20px; margin: 10px;"

st.markdown("<div style='display: flex;'>", unsafe_allow_html=True)

# Create two columns for layout with border and rounded corners
col1, col2 = st.columns([2, 2.9])
border_style = "border: 1px solid #dddddd; border-radius: 10px; padding: 10px;"

# Create placeholders for the character image and the camera frame inside the columns
character_placeholder = col1.empty()
frame_placeholder = col2.empty()




st.markdown("</div>", unsafe_allow_html=True)


# Path to the local image file with escaped backslashes
image_path= './images/rago.jpeg'

# Display the image
st.image(image_path, caption='Rago Chunyi', use_column_width=True, channels='BGR')

character_image_height = 500  # Adjust the height as needed

running = True  # Control variable for the loop


# Define frame size and frame rate

image_height = 322
frame_width = 500
frame_height = 500
frame_rate = 30



# Set the camera frame resolution
cap.set(3, frame_width)
cap.set(4, frame_height)



while running:
    ret, frame = cap.read()
    
    if not ret:
        st.error("Error: Unable to retrieve frames from the camera.")
        st.stop()
    frame=cv2.flip(frame,1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    data_aux = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            hand_data = []
            for landmark in hand_landmarks.landmark:
                hand_data.append(landmark.x - min(x_))
                hand_data.append(landmark.y - min(y_))

            data_aux.extend(hand_data)

    # Ensure data is always of length 84
    while len(data_aux) < 84:
        data_aux.extend([0, 0])  # append zero for undetected hand landmarks

    prediction = model.predict([np.asarray(data_aux)])
    predicted_character_image_path = labels_dict[int(prediction[0])]

    if predicted_character_image_path:
        # Load the predicted character image
        predicted_character_image = cv2.imread(predicted_character_image_path)

        # Resize the character image to match the desired height
        character_width = int((predicted_character_image.shape[1] / predicted_character_image.shape[0]) * character_image_height)
        predicted_character_image = cv2.resize(predicted_character_image, (character_width, character_image_height))

        # Create a white background
        white_background = np.ones((character_image_height, character_width, 3), dtype=np.uint8) * 255

        # Calculate the position to place the character image in the center of the white background
        x_offset = (white_background.shape[1] - predicted_character_image.shape[1]) // 2

        # Place the character image on the white background
        white_background[:, x_offset:x_offset + character_width] = predicted_character_image

        character_placeholder.image(white_background, channels="BGR", use_column_width=False, width=image_height)

    # Use st.image() for displaying the webcam feed
    frame_placeholder.image(frame, channels="BGR", use_column_width=True)


# Release the camera and close any OpenCV windows
release_camera()