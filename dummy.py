# import cv2

# # Replace with your actual RTSP URL
# rtsp_url = "rtsp://169.254.186.253/media/video1"

# # Initialize VideoCapture
# cap = cv2.VideoCapture(rtsp_url)

# # Check if the connection was successful
# if not cap.isOpened():
#     print("Cannot open RTSP stream. Check the RTSP URL and ensure the stream is active.")
#     exit()

# # Set desired frame width and height
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # Set desired FPS
# cap.set(cv2.CAP_PROP_FPS, 15)

# print("Successfully connected to the RTSP stream with optimized settings.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     cv2.imshow('Sony SRG-300SG Camera Stream', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



