import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from PIL import Image
import pyautogui
import keyboard

capture = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands()

# STEP 2: Create a HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

input_enabled = False

def toggle_input_activation():
    """Toggle the input activation on and off."""
    global input_enabled
    input_enabled = not input_enabled
    print(f"Input enabled: {input_enabled}")

keyboard.add_hotkey('ctrl+shift+a', toggle_input_activation)

def draw_landmarks_on_image(rgb_image, detection_result):
    if detection_result is None or not detection_result.hand_landmarks:
        return rgb_image

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def calculate_hand_means(rgb_image, detection_result):
    if detection_result is None or not detection_result.hand_landmarks:
        return None, None

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    x_right_list = []
    y_right_list = []
    x_left_list = []
    y_left_list = []

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        for l in hand_landmarks:
            if handedness[0].index == 0:  # Right hand
                x_right_list.append(l.x)
                y_right_list.append(l.y)
            else:  # Left hand
                x_left_list.append(l.x)
                y_left_list.append(l.y)

    height, width, _ = rgb_image.shape

    right_hand_mean = None
    left_hand_mean = None

    try:
        if x_right_list and y_right_list:
            x_right_mean = np.mean(x_right_list)
            y_right_mean = np.mean(y_right_list)
            right_x = int(x_right_mean * width)
            right_y = int(y_right_mean * height) - MARGIN
            right_hand_mean = (right_x, right_y)
    except Exception as e:
        print(f"Error processing right hand: {e}")

    try:
        if x_left_list and y_left_list:
            x_left_mean = np.mean(x_left_list)
            y_left_mean = np.mean(y_left_list)
            left_x = int(x_left_mean * width)
            left_y = int(y_left_mean * height) - MARGIN
            left_hand_mean = (left_x, left_y)
    except Exception as e:
        print(f"Error processing left hand: {e}")

    return right_hand_mean, left_hand_mean

def draw_hand_means(rgb_image, right_hand_mean, left_hand_mean):
    annotated_image = np.copy(rgb_image)

    if right_hand_mean:
        cv2.circle(annotated_image, right_hand_mean, 10, (0, 0, 255), 2)

    if left_hand_mean:
        cv2.circle(annotated_image, left_hand_mean, 10, (255, 0, 0), 2)

    if right_hand_mean and left_hand_mean:
        cv2.line(annotated_image, right_hand_mean, left_hand_mean, (0, 255, 0), 2)

        # Calculate distance and angle
        distance = calculate_distance(right_hand_mean, left_hand_mean)
        angle = calculate_angle(right_hand_mean, left_hand_mean)

        # Draw the distance and angle on the image
        draw_info(annotated_image, distance, angle)

    return annotated_image

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_angle(point1, point2):
    """Calculate the angle between two points relative to the horizontal axis."""
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle

def draw_info(image, distance, angle):
    """Draw the distance and angle on the top-left corner of the image."""
    text_dist = f"Distance: {distance:.2f}px"
    text_ang = f"Angle: {angle:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # White
    thickness = 1
    cv2.putText(image, text_dist, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(image, text_ang, (10, 50), font, font_scale, color, thickness, cv2.LINE_AA)

def approximate_angle(angle):
    """Approximate the angle to the nearest target angle."""
    target_angles = [0, 15, 30, 45, 60, -15, -30, -45, -60]
    closest_angle = min(target_angles, key=lambda x: abs(x - angle))
    return closest_angle

def move_mouse(left_hand_mean):
    """Move the mouse cursor using the left hand mean position."""
    if left_hand_mean:
        x, y = left_hand_mean
        screen_width, screen_height = pyautogui.size()
        target_x = int(x * screen_width)
        target_y = int(y * screen_height)
        pyautogui.moveTo(target_x, target_y)

while True:
    _, frame = capture.read()

    frame = cv2.flip(frame, 1)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    detection_result = detector.detect(image)

    pil_img = Image.new('RGB', (frame.shape[0], frame.shape[1]), color='black')
    test = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img)
    )

    annotated_image = draw_landmarks_on_image(test.numpy_view(), detection_result)

    right, left = calculate_hand_means(test.numpy_view(), detection_result)
    mean_image = draw_hand_means(test.numpy_view(), right, left)

    if right and left:
        slope = calculate_angle(right, left)
        approx_slope = approximate_angle(slope)

        if input_enabled:
            move_mouse(left)
    
    if approx_slope:
        pass

    cv2.imshow('og', frame)
    cv2.imshow('frame', annotated_image)

    try:
        cv2.imshow('mean', mean_image)
    except Exception as e:
        print(f"Error displaying mean image: {e}")

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()