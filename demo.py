# demo.py
"""
Demo script for the Image Editor project.
This script demonstrates main functionalities of the image editor.
"""

from ex5_helper import load_image, save_image, show_image
from image_editor import (
    RGB2grayscale,
    apply_kernel,
    blur_kernel,
    resize,
    rotate_90,
    find_patch_in_img,
)

INPUT_IMAGE_PATH = "examples/img2.jpg"
PATCH_IMAGE_PATH = "examples/img2 patch.jpg"
OUTPUT_GRAYSCALE_PATH = "examples/output_grayscale.png"
OUTPUT_BLUR_PATH = "examples/output_blur.png"
OUTPUT_RESIZE_PATH = "examples/output_resize.png"
OUTPUT_ROTATE_RIGHT_PATH = "examples/output_rotate_right.png"
OUTPUT_ROTATE_LEFT_PATH = "examples/output_rotate_left.png"


def demo_grayscale(input_path: str, output_path: str) -> None:
    img = load_image(input_path)
    gray_img = RGB2grayscale(img)
    save_image(gray_img, output_path)
    print(f"[GRAYSCALE] Output saved to {output_path}")
    show_image(gray_img)


def demo_blur(input_path: str, output_path: str, kernel_size: int = 10) -> None:
    img = load_image(input_path)
    gray_img = RGB2grayscale(img)
    kernel = blur_kernel(kernel_size)
    blurred_img = apply_kernel(gray_img, kernel)
    save_image(blurred_img, output_path)
    print(f"[BLUR] Output saved to {output_path}")
    show_image(blurred_img)



def demo_resize(input_path: str, output_path: str, scale: float = 0.5) -> None:
    img = load_image(input_path)
    gray_img = RGB2grayscale(img)
    new_height = int(len(img) * scale)
    new_width = int(len(img[0]) * scale)
    resized_img = resize(gray_img, new_height, new_width)
    save_image(resized_img, output_path)
    print(f"[RESIZE] Output saved to {output_path}")
    show_image(resized_img)


def demo_rotate(
    input_path: str,
    output_path_r: str,
    output_path_l: str,
) -> None:
    img = load_image(input_path)
    gray_img = RGB2grayscale(img)
    rotated_r = rotate_90(gray_img, "R")
    rotated_l = rotate_90(gray_img, "L")
    save_image(rotated_r, output_path_r)
    save_image(rotated_l, output_path_l)
    print(f"[ROTATE] Right saved to {output_path_r}")
    print(f"[ROTATE] Left saved to {output_path_l}")
    show_image(rotated_r)
    show_image(rotated_l)


def demo_find_patch(image_path: str, patch_path: str) -> None:
    img = load_image(image_path)
    patch = load_image(patch_path)
    gray_img = RGB2grayscale(img)
    gray_patch = RGB2grayscale(patch)
    locations = find_patch_in_img(gray_img, gray_patch)
    print("[FIND PATCH] Patch locations (multi-scale, rotations):")
    for angle, matches in locations.items():
        print(f"  Angle {angle}°: {matches}")


def run_demo() -> None:
    print("Starting Image Editor Demo...\n")
    demo_grayscale(INPUT_IMAGE_PATH, OUTPUT_GRAYSCALE_PATH)
    demo_blur(INPUT_IMAGE_PATH, OUTPUT_BLUR_PATH)
    demo_resize(INPUT_IMAGE_PATH, OUTPUT_RESIZE_PATH)
    demo_rotate(
        INPUT_IMAGE_PATH,
        OUTPUT_ROTATE_RIGHT_PATH,
        OUTPUT_ROTATE_LEFT_PATH,
    )
    demo_find_patch(INPUT_IMAGE_PATH, PATCH_IMAGE_PATH)
    print("\nDemo finished successfully!")


if __name__ == "__main__":
    run_demo()
