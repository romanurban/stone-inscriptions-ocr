import os
from object_selection import ObjectSelection

def process_images(directory):
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png')
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(supported_formats):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                # Initialize the ObjectSelection with the current file
                object_selector = ObjectSelection(file_path)
                
                # Run the object selection process
                object_selector.run()

# Example usage:
if __name__ == "__main__":
    # directory_path = 'dataset/preprocessing_test'
    # directory_path = 'dataset/timenote/Jaunciema_kapi/'
    directory_path = 'dataset/berlin-mitte/'
    process_images(directory_path)
