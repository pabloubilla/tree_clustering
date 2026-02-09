import os
import re
from PIL import Image
import imageio

def create_gif_from_images(input_dir, gif_name, output_path, duration=0.5):
    """
    Create a GIF from images in the specified directory.

    Parameters:
    input_dir (str): Directory containing the images.
    gif_name (str): The specific gif name to look for in the image filenames.
    output_path (str): Path to save the output GIF.
    duration (float): Duration between frames in the GIF (default is 0.5 seconds).
    """
    # Regex pattern to match the required files
    pattern = re.compile(r'(\d+)_{}.png'.format(gif_name))
    
    # List to store the image file paths
    image_files = []
    
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            image_files.append((int(match.group(1)), os.path.join(input_dir, filename)))

    # Sort image files based on the numerical part
    image_files.sort(key=lambda x: x[0])
    
    # Load images
    images = [Image.open(file[1]) for file in image_files]
    
    # Create and save the GIF
    imageio.mimsave(output_path, images, duration=duration)
    print(f'GIF saved to {output_path}')

if __name__ == '__main__':
    input_directory = os.path.join("server","gmm_error1.0","full_data","plots")
    gif_name = 't-SNE'  # The specific gif name
    output_gif_path = os.path.join(input_directory,"output.gif")
    create_gif_from_images(input_directory, gif_name, output_gif_path, duration=1400)

# Example usage:
# input_directory = 'path/to/your/images'
# gif_name = 'example'  # The specific gif name
# output_gif_path = 'path/to/save/your/output.gif'
# create_gif_from_images(input_directory, gif_name, output_gif_path, duration=0.5)
