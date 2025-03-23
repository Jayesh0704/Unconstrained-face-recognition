# import threading
# import time

# import cv2
# import numpy as np
# import torch
# import yaml
# from torchvision import transforms

# from face_alignment.alignment import norm_crop
# from face_detection.scrfd.detector import SCRFD
# from face_detection.yolov5_face.detector import Yolov5Face
# from face_recognition.arcface.model import iresnet_inference
# from face_recognition.arcface.utils import compare_encodings, read_features
# from face_tracking.tracker.byte_tracker import BYTETracker
# from face_tracking.tracker.visualize import plot_tracking

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Face detector (choose one)
# detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# # detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# # Face recognizer
# recognizer = iresnet_inference(
#     model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
# )

# # Load precomputed face features and names
# images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# # Mapping of face IDs to names
# id_face_mapping = {}

# # Data mapping for tracking information
# data_mapping = {
#     "raw_image": None,
#     "tracking_ids": [],
#     "detection_bboxes": [],
#     "detection_landmarks": [],
#     "tracking_bboxes": [],
# }

# def load_config(file_name):
#     """
#     Load a YAML configuration file.

#     Args:
#         file_name (str): The path to the YAML configuration file.

#     Returns:
#         dict: The loaded configuration as a dictionary.
#     """
#     with open(file_name, "r") as stream:
#         try:
#             return yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#             return {}

# def process_tracking(frame, detector, tracker, args, frame_id, fps):
#     """
#     Process tracking for a frame.

#     Args:
#         frame: The input frame.
#         detector: The face detector.
#         tracker: The object tracker.
#         args (dict): Tracking configuration parameters.
#         frame_id (int): The frame ID.
#         fps (float): Frames per second.

#     Returns:
#         numpy.ndarray: The processed tracking image.
#     """
#     # Face detection and tracking
#     outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

#     tracking_tlwhs = []
#     tracking_ids = []
#     tracking_scores = []
#     tracking_bboxes = []

#     if outputs is not None:
#         online_targets = tracker.update(
#             outputs, [img_info["height"], img_info["width"]], (128, 128)
#         )

#         for t in online_targets:
#             tlwh = t.tlwh
#             tid = t.track_id
#             vertical = tlwh[2] / tlwh[3] > args.get("aspect_ratio_thresh", 1.6)
#             if tlwh[2] * tlwh[3] > args.get("min_box_area", 100) and not vertical:
#                 x1, y1, w, h = tlwh
#                 tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
#                 tracking_tlwhs.append(tlwh)
#                 tracking_ids.append(tid)
#                 tracking_scores.append(t.score)

#         tracking_image = plot_tracking(
#             img_info["raw_img"],
#             tracking_tlwhs,
#             tracking_ids,
#             names=id_face_mapping,
#             frame_id=frame_id + 1,
#             fps=fps,
#         )
#     else:
#         tracking_image = img_info["raw_img"]

#     data_mapping["raw_image"] = img_info["raw_img"]
#     data_mapping["detection_bboxes"] = bboxes
#     data_mapping["detection_landmarks"] = landmarks
#     data_mapping["tracking_ids"] = tracking_ids
#     data_mapping["tracking_bboxes"] = tracking_bboxes

#     return tracking_image

# @torch.no_grad()
# def get_feature(face_image):
#     """
#     Extract features from a face image.

#     Args:
#         face_image: The input face image.

#     Returns:
#         numpy.ndarray: The extracted features.
#     """
#     face_preprocess = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Resize((112, 112)),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ]
#     )

#     # Convert to RGB
#     face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

#     # Preprocess image
#     face_image = face_preprocess(face_image).unsqueeze(0).to(device)

#     # Inference to get feature
#     emb_img_face = recognizer(face_image).cpu().numpy()

#     # Normalize the embedding
#     images_emb = emb_img_face / np.linalg.norm(emb_img_face)

#     return images_emb

# def recognition(face_image):
#     """
#     Recognize a face image.

#     Args:
#         face_image: The input face image.

#     Returns:
#         tuple: A tuple containing the recognition score and name.
#     """
#     # Get feature from face
#     query_emb = get_feature(face_image)

#     score, id_min = compare_encodings(query_emb, images_embs)
#     name = images_names[id_min]
#     score = score[0]

#     return score, name

# def mapping_bbox(box1, box2):
#     """
#     Calculate the Intersection over Union (IoU) between two bounding boxes.

#     Args:
#         box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
#         box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

#     Returns:
#         float: The IoU score.
#     """
#     # Calculate the intersection area
#     x_min_inter = max(box1[0], box2[0])
#     y_min_inter = max(box1[1], box2[1])
#     x_max_inter = min(box1[2], box2[2])
#     y_max_inter = min(box1[3], box2[3])

#     intersection_area = max(0, x_max_inter - x_min_inter) * max(
#         0, y_max_inter - y_min_inter
#     )

#     # Calculate the area of each bounding box
#     area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     # Calculate the union area
#     union_area = area_box1 + area_box2 - intersection_area

#     if union_area == 0:
#         return 0

#     # Calculate IoU
#     iou = intersection_area / union_area

#     return iou

# def tracking(detector, args, frame_grabber):
#     """
#     Face tracking in a separate thread.

#     Args:
#         detector: The face detector.
#         args (dict): Tracking configuration parameters.
#         frame_grabber: Instance of FrameGrabber to get frames.
#     """
#     # Initialize variables for measuring frame rate
#     start_time = time.time_ns()
#     frame_count = 0
#     fps = 0.0

#     # Initialize a tracker and a timer
#     tracker = BYTETracker(args=args, frame_rate=30)
#     frame_id = 0

#     while True:
#         img = frame_grabber.get_frame()
#         if img is None:
#             continue

#         tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)

#         # Calculate and display the frame rate
#         frame_count += 1
#         if frame_count >= 30:
#             elapsed_time = time.time_ns() - start_time
#             fps = 1e9 * frame_count / elapsed_time
#             frame_count = 0
#             start_time = time.time_ns()

#         cv2.imshow("Face Recognition", tracking_image)

#         # Check for user exit input
#         ch = cv2.waitKey(1)
#         if ch == 27 or ch == ord("q") or ch == ord("Q"):
#             frame_grabber.stop()
#             break

#         frame_id += 1

#     cv2.destroyAllWindows()

# class FrameGrabber(threading.Thread):
#     """
#     A separate thread to continuously grab frames from the RTSP stream.
#     Keeps only the latest frame to minimize latency.
#     """
#     def __init__(self, rtsp_url, width=640, height=480, fps=15):
#         super(FrameGrabber, self).__init__()
#         self.rtsp_url = rtsp_url
#         self.cap = cv2.VideoCapture(rtsp_url)
#         if not self.cap.isOpened():
#             print("Cannot open RTSP stream. Check the RTSP URL and ensure the stream is active.")
#             raise ValueError("Cannot open RTSP stream.")
#         # Set desired frame width and height
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#         # Set desired FPS (Note: May not work with all cameras)
#         self.cap.set(cv2.CAP_PROP_FPS, fps)

#         self.latest_frame = None
#         self.stopped = False
#         self.lock = threading.Lock()

#     def run(self):
#         while not self.stopped:
#             ret, frame = self.cap.read()
#             if not ret:
#                 continue
#             with self.lock:
#                 self.latest_frame = frame

#     def get_frame(self):
#         with self.lock:
#             return self.latest_frame.copy() if self.latest_frame is not None else None

#     def stop(self):
#         self.stopped = True
#         self.cap.release()

# def recognize():
#     """Face recognition in a separate thread."""
#     while True:
#         raw_image = data_mapping["raw_image"]
#         detection_landmarks = data_mapping["detection_landmarks"]
#         detection_bboxes = data_mapping["detection_bboxes"]
#         tracking_ids = data_mapping["tracking_ids"]
#         tracking_bboxes = data_mapping["tracking_bboxes"]

#         if raw_image is None:
#             continue

#         for i in range(len(tracking_bboxes)):
#             for j in range(len(detection_bboxes)):
#                 mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
#                 if mapping_score > 0.9:
#                     face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])

#                     score, name = recognition(face_image=face_alignment)
#                     if name is not None:
#                         if score < 0.25:
#                             caption = "UN_KNOWN"
#                         else:
#                             caption = f"{name}:{score:.2f}"
#                     else:
#                         caption = "UN_KNOWN"

#                     id_face_mapping[tracking_ids[i]] = caption

#                     # Remove the matched detection to prevent duplicate processing
#                     detection_bboxes = np.delete(detection_bboxes, j, axis=0)
#                     detection_landmarks = np.delete(detection_landmarks, j, axis=0)

#                     break

#         if not tracking_bboxes:
#             print("Waiting for a person...")

#         # Small sleep to prevent high CPU usage
#         time.sleep(0.01)

# def main():
#     """Main function to start frame grabbing, tracking, and recognition threads."""
#     file_name = "./face_tracking/config/config_tracking.yaml"
#     config_tracking = load_config(file_name)

#     # Define the RTSP URL for the Sony SRG-300SE camera
#     rtsp_url = "rtsp://169.254.33.225/media/video1"

#     try:
#         # Start frame grabbing thread
#         frame_grabber = FrameGrabber(rtsp_url=rtsp_url, width=640, height=480, fps=15)
#         frame_grabber.start()

#         # Start tracking thread
#         thread_track = threading.Thread(
#             target=tracking,
#             args=(
#                 detector,
#                 config_tracking,
#                 frame_grabber,  # Pass the FrameGrabber instance
#             ),
#         )
#         thread_track.start()

#         # Start recognition thread
#         thread_recognize = threading.Thread(target=recognize)
#         thread_recognize.start()

#         # Join threads
#         thread_track.join()
#         thread_recognize.join()
#         frame_grabber.join()

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         if 'frame_grabber' in locals():
#             frame_grabber.stop()

# if __name__ == "__main__":
#     main()















#FOR NORMAL CAM
import threading
import time

import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector (choose one)
# detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# Face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

# Load precomputed face features and names
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# Mapping of face IDs to names
id_face_mapping = {}

# Data mapping for tracking information
data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
}


def load_config(file_name):
    """
    Load a YAML configuration file.

    Args:
        file_name (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def process_tracking(frame, detector, tracker, args, frame_id, fps):
    """
    Process tracking for a frame.

    Args:
        frame: The input frame.
        detector: The face detector.
        tracker: The object tracker.
        args (dict): Tracking configuration parameters.
        frame_id (int): The frame ID.
        fps (float): Frames per second.

    Returns:
        numpy.ndarray: The processed tracking image.
    """
    # Face detection and tracking
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )

        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)

        tracking_image = plot_tracking(
            img_info["raw_img"],
            tracking_tlwhs,
            tracking_ids,
            names=id_face_mapping,
            frame_id=frame_id + 1,
            fps=fps,
        )
    else:
        tracking_image = img_info["raw_img"]

    data_mapping["raw_image"] = img_info["raw_img"]
    data_mapping["detection_bboxes"] = bboxes
    data_mapping["detection_landmarks"] = landmarks
    data_mapping["tracking_ids"] = tracking_ids
    data_mapping["tracking_bboxes"] = tracking_bboxes

    return tracking_image


@torch.no_grad()
def get_feature(face_image):
    """
    Extract features from a face image.

    Args:
        face_image: The input face image.

    Returns:
        numpy.ndarray: The extracted features.
    """
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocess image (BGR)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Inference to get feature
    emb_img_face = recognizer(face_image).cpu().numpy()

    # Convert to array
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    return images_emb


def recognition(face_image):
    """
    Recognize a face image.

    Args:
        face_image: The input face image.

    Returns:
        tuple: A tuple containing the recognition score and name.
    """
    # Get feature from face
    query_emb = get_feature(face_image)

    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]

    return score, name


def mapping_bbox(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
        box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

    Returns:
        float: The IoU score.
    """
    # Calculate the intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def tracking(detector, args):
    """
    Face tracking in a separate thread.

    Args:
        detector: The face detector.
        args (dict): Tracking configuration parameters.
    """
    # Initialize variables for measuring frame rate
    start_time = time.time_ns()
    frame_count = 0
    fps = -1

    # Initialize a tracker and a timer
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()

        tracking_image = process_tracking(img, detector, tracker, args, frame_id, fps)

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()

        cv2.imshow("Face Recognition", tracking_image)

        # Check for user exit input
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def recognize():
    """Face recognition in a separate thread."""
    while True:
        raw_image = data_mapping["raw_image"]
        detection_landmarks = data_mapping["detection_landmarks"]
        detection_bboxes = data_mapping["detection_bboxes"]
        tracking_ids = data_mapping["tracking_ids"]
        tracking_bboxes = data_mapping["tracking_bboxes"]

        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                if mapping_score > 0.9:
                    face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])

                    score, name = recognition(face_image=face_alignment)
                    if name is not None:
                        if score < 0.25:
                            caption = "UN_KNOWN"
                        else:
                            caption = f"{name}:{score:.2f}"

                    id_face_mapping[tracking_ids[i]] = caption

                    detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)

                    break

        if tracking_bboxes == []:
            print("Waiting for a person...")


def main():
    """Main function to start face tracking and recognition threads."""
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    # Start tracking thread
    thread_track = threading.Thread(
        target=tracking,
        args=(
            detector,
            config_tracking,
        ),
    )
    thread_track.start()

    # Start recognition thread
    thread_recognize = threading.Thread(target=recognize)
    thread_recognize.start()


if __name__ == "__main__":
    main()