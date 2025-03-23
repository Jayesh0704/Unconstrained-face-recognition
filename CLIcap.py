import threading
from image_capture import capture_images

def message_callback(message, type='info'):
    """Prints messages to the console. Customize as needed."""
    print(f"[{type.upper()}] {message}")

def main():
    # Get user input for ID and Name
    id_no = input("Enter your ID: ")
    name = input("Enter your Name: ")
    
    # Create an event to allow stopping capture if needed
    stop_event = threading.Event()
    
    # Start capturing images
    try:
        capture_images(id_no, name, message_callback, stop_event)
    except KeyboardInterrupt:
        stop_event.set()
        print("Capture interrupted by user.")

if __name__ == "__main__":
    main()
