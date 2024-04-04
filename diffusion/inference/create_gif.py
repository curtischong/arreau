import os
import glob
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_gif(src_img_dir: str, output_file: str, duration_ms: int = 100):
    image_files = glob.glob(os.path.join(src_img_dir, "*.png"))

    # Sort the image files by creation time (ascending order)
    image_files.sort(key=os.path.getctime)

    # Open the first image
    first_image = Image.open(image_files[0])

    # Iterate over the remaining image files
    frames = []
    for image_file in image_files[1:]:
        try:
            image = Image.open(image_file)
            frames.append(image)
        except (IOError, OSError):
            print(f"Skipping invalid image file: {image_file}")

    first_image.save(
        output_file, save_all=True, append_images=frames, duration=duration_ms, loop=0
    )
    print(f"GIF generated successfully: {output_file}")


if __name__ == "__main__":
    directory = "out"
    output_file = "out/crystal.gif"

    generate_gif(directory, output_file)
