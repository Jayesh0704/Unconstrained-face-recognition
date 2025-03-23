import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewFolderHandler(FileSystemEventHandler):
    def __init__(self, script_path, backup_dir, faces_save_dir, features_path):
        self.script_path = script_path
        self.backup_dir = backup_dir
        self.faces_save_dir = faces_save_dir
        self.features_path = features_path

    def on_created(self, event):
        # Check if a new folder was created
        if event.is_directory:
            print(f"New folder detected: {event.src_path}")
            
            # Run the add_person.py script
            subprocess.run([
                "python", self.script_path,
                "--backup-dir", self.backup_dir,
                "--add-persons-dir", event.src_path,
                "--faces-save-dir", self.faces_save_dir,
                "--features-path", self.features_path
            ])

def main():
    # Define paths and directories
    dataset_dir = "datasets/new_persons"
    script_path = "add_person.py"
    backup_dir = "datasets/backup"
    faces_save_dir = "datasets/data"
    features_path = "datasets/face_features/feature"

    # Create an event handler
    event_handler = NewFolderHandler(script_path, backup_dir, faces_save_dir, features_path)

    # Create an observer and schedule it to monitor the dataset directory
    observer = Observer()
    observer.schedule(event_handler, path=dataset_dir, recursive=False)

    # Start monitoring
    observer.start()
    print(f"Monitoring {dataset_dir} for new folders...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    main()
