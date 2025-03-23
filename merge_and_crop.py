import os
from PIL import Image

def merge_and_pad_images(input_folder, output_file, grid_size=(2,2), target_size=(300, 300), margin=10):
    # Get image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Ensure enough images
    if len(image_files) < grid_size[0] * grid_size[1]:
        raise ValueError(f"Not enough images. Need at least {grid_size[0] * grid_size[1]} images.")
    
    # Open and resize images with consistent margins
    images = []
    for img_name in image_files[:grid_size[0] * grid_size[1]]:
        img_path = os.path.join(input_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # Resize image to fit within target size while maintaining aspect ratio
        img.thumbnail((target_size[0] - 2*margin, target_size[1] - 2*margin), Image.LANCZOS)
        
        # Create a white background with the target size including margins
        img_width, img_height = img.size
        background = Image.new('RGB', target_size, (255, 255, 255))
        
        # Calculate the offset to center the image with margins
        offset_x = (target_size[0] - img_width) // 2
        offset_y = (target_size[1] - img_height) // 2
        
        # Paste the resized image onto the center of the background
        background.paste(img, (offset_x, offset_y))
        
        images.append(background)
    
    # Create the collage
    collage_width = target_size[0] * grid_size[1]
    collage_height = target_size[1] * grid_size[0]
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
    
    # Paste images into grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            index = i * grid_size[1] + j
            x = j * target_size[0]
            y = i * target_size[1]
            collage.paste(images[index], (x, y))
    
    # Save the collage
    collage.save(output_file)
    print(f"Merged and padded image saved to {output_file}")

# Example usage

if __name__ == "__main__":
    input_folder = "D:\SOP LOP DOP\Face recog\Face-Recognition-advance-main\occlusion-based-images"
    output_file = "D:\SOP LOP DOP\Face recog\Face-Recognition-advance-main\occlusion-based-images-output\merged_and_padded_image.jpg"
    
    merge_and_pad_images(input_folder, output_file)