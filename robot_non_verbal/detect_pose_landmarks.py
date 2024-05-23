# Fonte: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=pt-br

# pip install mediapipe
# pip install opencv-python-headless
# pip install opencv-python
# wget -O tasks/pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
# wget -q -O input_images/input_pose.jpg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg

# STEP 1: Import the necessary modules.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.utils_pose import draw_landmarks_on_image

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='tasks/pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("input_images/input_pose.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imwrite("output_images/output_pose2.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)) 
