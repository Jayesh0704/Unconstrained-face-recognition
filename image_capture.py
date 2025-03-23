# import cv2
# import os
# import time
# import shutil

# def capture_images(id_no, name, message_callback, stop_event, num_images=100, delay=0.1):
#     """
#     Captures images from the webcam, saves them to a specified folder,
#     and duplicates them to reach a total of 200 images.

#     Parameters:
#     - id_no (str): The ID number of the user.
#     - name (str): The name of the user.
#     - message_callback (function): Function to send log messages to the GUI.
#     - stop_event (threading.Event): Event to signal stopping the capture.
#     - num_images (int): Number of images to capture (half of the total desired count).
#     - delay (float): Delay between captures in seconds.
#     """
#     parent_folder = "captured_images"
#     # Create the parent folder if it does not exist
#     if not os.path.exists(parent_folder):
#         os.makedirs(parent_folder)
#         message_callback(f"Created parent folder '{parent_folder}'.\n")

#     # Create the folder name using ID NO and Name
#     folder_name = f"{id_no} - {name}"
#     save_folder = os.path.join(parent_folder, folder_name)

#     # Check if the folder already exists to prevent duplicates
#     if os.path.exists(save_folder):
#         message_callback(f"The folder '{folder_name}' already exists. Please use a unique ID and Name.\n", type='error')
#         return

#     # Create the user-specific folder
#     os.makedirs(save_folder, exist_ok=True)
#     message_callback(f"Created folder '{folder_name}'. Starting image capture.\n")

#     # Open the camera
#     cap = cv2.VideoCapture(0)  # 0 is the default camera

#     # Check if the camera opened successfully
#     if not cap.isOpened():
#         message_callback("Error: Could not open the camera.\n", type='error')
#         return

#     message_callback("Starting image capture. Press 'q' in the OpenCV window to quit early...\n")

#     for i in range(1, num_images + 1):
#         if stop_event.is_set():
#             message_callback("Image capture stopped by user.\n", type='info')
#             break

#         ret, frame = cap.read()

#         if not ret:
#             message_callback(f"Failed to grab frame {i}.\n", type='error')
#             break

#         # Show the frame on screen
#         cv2.imshow("Capturing", frame)

#         # Save the frame in color with the folder name as part of the image name
#         image_path = os.path.join(save_folder, f"{id_no} - {name}_{i}.jpg")
#         cv2.imwrite(image_path, frame)
#         message_callback(f"Saved {image_path}\n", type='info')

#         # Wait for a short period before capturing the next frame
#         key = cv2.waitKey(int(delay * 1000))  # delay in milliseconds
#         if key & 0xFF == ord('q'):
#             message_callback("Early exit requested by pressing 'q'.\n", type='info')
#             break

#     # Release the camera and close windows
#     cap.release()
#     cv2.destroyAllWindows()

#     message_callback("Finished capturing initial images. Starting duplication to reach 200 images.\n")

#     # Duplicate images to reach 200
#     for i in range(1, num_images + 1):
#         original_image_path = os.path.join(save_folder, f"{id_no} - {name}_{i}.jpg")
#         duplicate_image_path = os.path.join(save_folder, f"{id_no} - {name}_{i + num_images}.jpg")
#         shutil.copyfile(original_image_path, duplicate_image_path)
#         message_callback(f"Duplicated {original_image_path} to {duplicate_image_path}\n", type='info')

#     message_callback("Completed duplication. Total of 200 images available.\n")


import cv2
import os
import time
import shutil
import re

def capture_images(id_no, name, message_callback, stop_event, num_images=100, delay=0.1):
    """
    Captures images from the webcam, saves them to a specified folder,
    and duplicates them to reach a total of 200 images.

    Parameters:
    - id_no (str): The ID number of the user.
    - name (str): The name of the user.
    - message_callback (function): Function to send log messages to the GUI.
    - stop_event (threading.Event): Event to signal stopping the capture.
    - num_images (int): Number of images to capture (half of the total desired count).
    - delay (float): Delay between captures in seconds.
    """
    parent_folder = "captured_images"
    # Create the parent folder if it does not exist
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
        message_callback(f"Created parent folder '{parent_folder}'.\n")

    # Remove invalid characters from id_no and name for folder compatibility
    id_no = re.sub(r'[<>:"/\\|?*]', '', id_no)
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    
    # Create the folder name using sanitized ID NO and Name
    folder_name = f"{id_no} - {name}"
    save_folder = os.path.join(parent_folder, folder_name)

    # Check if the folder already exists to prevent duplicates
    if os.path.exists(save_folder):
        message_callback(f"The folder '{folder_name}' already exists. Please use a unique ID and Name.\n", type='error')
        return

    # Create the user-specific folder
    os.makedirs(save_folder, exist_ok=True)
    message_callback(f"Created folder '{folder_name}'. Starting image capture.\n")

    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    # Check if the camera opened successfully
    if not cap.isOpened():
        message_callback("Error: Could not open the camera.\n", type='error')
        return

    message_callback("Starting image capture. Press 'q' in the OpenCV window to quit early...\n")

    for i in range(1, num_images + 1):
        if stop_event.is_set():
            message_callback("Image capture stopped by user.\n", type='info')
            break

        ret, frame = cap.read()

        if not ret:
            message_callback(f"Failed to grab frame {i}.\n", type='error')
            break

        # Show the frame on screen
        cv2.imshow("Capturing", frame)

        # Save the frame in color with the folder name as part of the image name
        image_path = os.path.join(save_folder, f"{id_no} - {name}_{i}.jpg")
        cv2.imwrite(image_path, frame)
        message_callback(f"Saved {image_path}\n", type='info')

        # Wait for a short period before capturing the next frame
        key = cv2.waitKey(int(delay * 1000))  # delay in milliseconds
        if key & 0xFF == ord('q'):
            message_callback("Early exit requested by pressing 'q'.\n", type='info')
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    message_callback("Finished capturing initial images. Starting duplication to reach 200 images.\n")

    # Duplicate images to reach 200
    for i in range(1, num_images + 1):
        original_image_path = os.path.join(save_folder, f"{id_no} - {name}_{i}.jpg")
        duplicate_image_path = os.path.join(save_folder, f"{id_no} - {name}_{i + num_images}.jpg")
        shutil.copyfile(original_image_path, duplicate_image_path)
        message_callback(f"Duplicated {original_image_path} to {duplicate_image_path}\n", type='info')

    message_callback("Completed duplication. Total of 200 images available.\n")
