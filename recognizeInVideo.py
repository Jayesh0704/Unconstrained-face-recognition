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
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# Face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

# Load precomputed face features and names
images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# Global mapping of face tracking IDs to recognized names
recognized_faces = {}

def load_config(file_name):
    """Load a YAML configuration file."""
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

@torch.no_grad()
def get_feature(face_image):
    """Extract features from a face image."""
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    emb_img_face = recognizer(face_image).cpu().numpy()
    return emb_img_face / np.linalg.norm(emb_img_face)

def recognize_face(face_image, confidence_threshold=0.30):
    """
    Recognize a face and return name and confidence.
    
    Args:
        face_image: Aligned face image
        confidence_threshold: Minimum confidence to consider a valid recognition
    
    Returns:
        tuple: (name, confidence score)
    """
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    confidence = score[0]
    
    if confidence > confidence_threshold:
        return name, confidence
    return "Not in Dataset", confidence

def process_tracking(frame, detector, tracker, args, frame_id, fps):
    """Process tracking for a frame."""
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)

    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )

        for i, t in enumerate(online_targets):
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)

        # Use recognized names for tracking display
        tracking_image = plot_tracking(
            img_info["raw_img"],
            tracking_tlwhs,
            tracking_ids,
            names={tid: recognized_faces.get(tid, f"ID {tid}") for tid in tracking_ids},
            frame_id=frame_id + 1,
            fps=fps
        )
    else:
        tracking_image = img_info["raw_img"]

    return tracking_image, img_info, bboxes, landmarks, tracking_ids

def tracking(detector, args, video_path, output_path):
    """Face tracking and recognition for a video file."""
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0
    fps_calc_interval = 30
    start_time = time.time_ns()
    frame_count = 0
    current_fps = -1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracking_image, img_info, bboxes, landmarks, tracking_ids = process_tracking(
            frame, detector, tracker, args, frame_id, current_fps
        )

        # Process face recognition
        for idx, (bbox, landmark, tracking_id) in enumerate(
            zip(bboxes, landmarks, tracking_ids)
        ):
            # Align face
            face_image = norm_crop(img_info["raw_img"], landmark)
            
            # Recognize face
            name, confidence = recognize_face(face_image)
            
            # Update recognized faces mapping
            if confidence > 0.3:
                recognized_faces[tracking_id] = f"{name} ({confidence:.2f})"
                print(f"Tracking ID {tracking_id}: {name} (Confidence: {confidence:.2f})")

        out.write(tracking_image)

        # Optional: Display processing
        cv2.imshow("Face Recognition", tracking_image)
        if cv2.waitKey(1) in [27, ord("q"), ord("Q")]:
            break

        # FPS calculation
        frame_count += 1
        if frame_count >= fps_calc_interval:
            current_fps = 1e9 * frame_count / (time.time_ns() - start_time)
            frame_count = 0
            start_time = time.time_ns()

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    """Main function to start face tracking on a video file."""
    config_file = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(config_file)

    video_path = "video/input.mp4"
    output_path = "output.mp4"

    tracking(detector, config_tracking, video_path, output_path)

if __name__ == "__main__":
    main()
