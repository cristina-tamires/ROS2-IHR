# Fonte: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer?hl=pt-br

# pip install -q mediapipe
# wget -q -O tasks/gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
# wget -q -O input_images/thumbs_up.jpg https://cdn.pixabay.com/photo/2017/08/16/22/27/thumbs-up-2649310_1280.jpg
# wget -q -O input_images/thumbs_down.jpg https://cdn.pixabay.com/photo/2021/10/26/14/07/thumbs-down-6744094_1280.jpg
# wget -q -O input_images/pointing_up.jpg https://cdn.pixabay.com/photo/2018/01/14/23/08/idea-3082824_1280.jpg


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.utils_gesture import display_batch_of_images_with_gestures_and_hand_landmarks

# STEP 2: Create an GestureRecognizer object.
base_options = python.BaseOptions(model_asset_path='tasks/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

IMAGE_FILENAMES = ["input_images/thumbs_up.jpg","input_images/thumbs_down.jpg","input_images/pointing_up.jpg"]

images = []
results = []
for image_file_name in IMAGE_FILENAMES:
  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(image_file_name)

  # STEP 4: Recognize gestures in the input image.
  recognition_result = recognizer.recognize(image)

  # STEP 5: Process the result. In this case, visualize it.
  images.append(image)
#   print (recognition_result.gestures)

  top_gesture = recognition_result.gestures[0][0]
  hand_landmarks = recognition_result.hand_landmarks
  results.append((top_gesture, hand_landmarks))

display_batch_of_images_with_gestures_and_hand_landmarks(images, results)