import cv2
import os

# Define the parent folders where images will be saved
parent_folder_captured_images = "captured_images"
parent_folder_datasets = "datasets/new_persons"

# Create the parent folders if they do not exist
if not os.path.exists(parent_folder_captured_images):
    os.makedirs(parent_folder_captured_images)

if not os.path.exists(parent_folder_datasets):
    os.makedirs(parent_folder_datasets)

# Prompt user for a folder name to save the images inside both parent folders
folder_name = input("Enter a name for the folder to save images: ")

# Create the path for the user-specified folder inside each parent folder
save_folder_captured_images = os.path.join(parent_folder_captured_images, folder_name)
save_folder_datasets = os.path.join(parent_folder_datasets, folder_name)

if not os.path.exists(save_folder_captured_images):
    os.makedirs(save_folder_captured_images)

if not os.path.exists(save_folder_datasets):
    os.makedirs(save_folder_datasets)

# Open the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Number of images to capture
num_images = 200

print("Press 'q' to quit early...")

for i in range(num_images):
    ret, frame = cap.read()

    if not ret:
        print(f"Failed to grab frame {i+1}")
        break

    # Show the frame on screen
    cv2.imshow("Capturing", frame)

    # Save the frame in color to both folders
    image_path_captured_images = os.path.join(save_folder_captured_images, f"{folder_name}_{i+1}.jpg")
    image_path_datasets = os.path.join(save_folder_datasets, f"{folder_name}_{i+1}.jpg")

    cv2.imwrite(image_path_captured_images, frame)
    cv2.imwrite(image_path_datasets, frame)

    print(f"Saved {image_path_captured_images}")
    print(f"Saved {image_path_datasets}")

    # Wait for a short period before capturing the next frame
    key = cv2.waitKey(100)  # 100 ms delay between captures
    if key & 0xFF == ord('q'):
        print("Early exit requested.")
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

print("Finished capturing images.")


