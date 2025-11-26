#################################################################
# FILE : image_editor.py
# WRITER : Yaacov Pinkert
# EXERCISE : intro2cs1 ex5 2024
# DESCRIPTION: Image Processing Functions
#################################################################
import copy
import math
from ex5_helper import *
from typing import Optional, Dict, Tuple, Any


PERCENT_RED = 0.299
PERCENT_GREEN = 0.587
PERCENT_BLUE = 0.114


def list_dimensions(i_list: list, num: int):
    """Return dimensions of a list.

    If num == 2: (rows, cols) of 2D list.
    If num == 3: (rows, cols, depth) of 3D list.
    """
    if num == 2:
        return len(i_list), len(i_list[0])
    if num == 3:
        return len(i_list), len(i_list[0]), len(i_list[0][0])
    # logic of the exercise never calls with other values
    return None


def relative_location(loc: float,
                      destination_dim: int,
                      origin_dim: int) -> float:
    """Map coordinate from destination size to origin size, linearly."""
    return (loc / destination_dim) * origin_dim


def init_image(height: int, width: int) -> SingleChannelImage:
    """Create 2D image (height x width) filled with zeros."""
    return [[0] * width for _ in range(height)]


def formula_interpolation(delta_y: float, delta_x: float,
                          a: int, b: int, c: int, d: int) -> int:
    """Bilinear interpolation of pixel from four neighbors a, b, c, d."""
    new_pixel = (
        a * (1 - delta_x) * (1 - delta_y)
        + b * delta_y * (1 - delta_x)
        + c * delta_x * (1 - delta_y)
        + d * delta_x * delta_y
    )
    return round(new_pixel)


def formula_rgb_to_gray(red: int, green: int, blue: int) -> int:
    """Convert RGB to gray using fixed coefficients."""
    gray = red * PERCENT_RED + green * PERCENT_GREEN + blue * PERCENT_BLUE
    return round(gray)


def formula_kernel(imagey: SingleChannelImage,
                   row_loc: int,
                   column_loc: int,
                   kernely: Kernel) -> int:
    """Apply convolution kernel at a specific position."""
    rows_img = len(imagey) - 1
    columns_img = len(imagey[0]) - 1
    kernel_size = len(kernely)
    deviation = kernel_size // 2
    start_row = row_loc - deviation
    start_column = column_loc - deviation

    cell_sum = 0.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            img_row = start_row + i
            img_col = start_column + j
            if img_row < 0 or img_col < 0:
                cell_sum += imagey[row_loc][column_loc] * kernely[i][j]
            elif img_row > rows_img or img_col > columns_img:
                cell_sum += imagey[row_loc][column_loc] * kernely[i][j]
            else:
                cell_sum += imagey[img_row][img_col] * kernely[i][j]

    cell_sum = round(cell_sum)
    if cell_sum < 0:
        return 0
    if cell_sum > 255:
        return 255
    return cell_sum


def separate_channels(image: ColoredImage) -> List[SingleChannelImage]:
    """Separate colored image into list of single-channel images."""
    rows, columns, channels = list_dimensions(image, 3)
    final_list: List[SingleChannelImage] = []

    for c in range(channels):
        channel_list: SingleChannelImage = []
        for i in range(rows):
            row_list = [image[i][j][c] for j in range(columns)]
            channel_list.append(row_list)
        final_list.append(channel_list)
    return final_list


def combine_channels(channels: List[SingleChannelImage]) -> ColoredImage:
    """Combine list of channels into a colored image."""
    channel_count, rows, columns = list_dimensions(channels, 3)
    final_list: ColoredImage = []

    for i in range(rows):
        row: List[Pixel] = []
        for j in range(columns):
            pixel = [channels[c][i][j] for c in range(channel_count)]
            row.append(pixel)
        final_list.append(row)
    return final_list


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """Convert colored image to grayscale."""
    final_list: SingleChannelImage = []
    for row in colored_image:
        gray_row: List[int] = []
        for pixel in row:
            new_cell = formula_rgb_to_gray(pixel[0], pixel[1], pixel[2])
            gray_row.append(new_cell)
        final_list.append(gray_row)
    return final_list


def blur_kernel(size: int) -> Kernel:
    """Create blur kernel of given size."""
    value = 1 / (size ** 2)
    return [[value for _ in range(size)] for _ in range(size)]


def apply_kernel(image: SingleChannelImage,
                 kernel: Kernel) -> SingleChannelImage:
    """Apply kernel to 2D single-channel image."""
    rows, columns = list_dimensions(image, 2)
    new_image = init_image(rows, columns)

    for i in range(rows):
        for j in range(columns):
            new_image[i][j] = formula_kernel(image, i, j, kernel)
    return new_image


def bilinear_interpolation(image: SingleChannelImage,
                           y: float,
                           x: float) -> int:
    """Bilinear interpolation for non-integer position (y, x)."""
    y_a = math.floor(y)
    x_a = math.floor(x)
    delta_y = y - y_a
    delta_x = x - x_a

    pix_a = image[y_a][x_a]
    pix_b = image[y_a + 1][x_a] if y_a + 1 < len(image) else pix_a
    pix_c = image[y_a][x_a + 1] if x_a + 1 < len(image[0]) else pix_a
    pix_d = pix_a if (pix_b == pix_a or pix_c == pix_a) else image[y_a + 1][x_a + 1]

    return formula_interpolation(delta_y, delta_x, pix_a, pix_b, pix_c, pix_d)


def resize(image: SingleChannelImage,
           new_height: int,
           new_width: int) -> SingleChannelImage:
    """Resize single-channel image using bilinear interpolation."""
    new_image = init_image(new_height, new_width)
    rows_img = len(image) - 1
    columns_img = len(image[0]) - 1

    # corners
    new_image[0][0] = image[0][0]
    new_image[0][new_width - 1] = image[0][columns_img]
    new_image[new_height - 1][0] = image[rows_img][0]
    new_image[new_height - 1][new_width - 1] = image[rows_img][columns_img]

    for i in range(new_height):
        for j in range(new_width):
            if (
                (i == 0 and j == 0)
                or (i == 0 and j == new_width - 1)
                or (i == new_height - 1 and j == 0)
                or (i == new_height - 1 and j == new_width - 1)
            ):
                continue
            rel_loc_y = relative_location(i, new_height - 1, rows_img)
            rel_loc_x = relative_location(j, new_width - 1, columns_img)
            new_image[i][j] = bilinear_interpolation(image, rel_loc_y, rel_loc_x)
    return new_image


def rotate_90(image: Image, direction: str) -> Image:
    """Rotate image 90 degrees to the right (R) or left (L)."""
    rows_origin, columns_origin = list_dimensions(image, 2)
    new_image: Image = []

    if direction == "R":
        for j in range(columns_origin):
            new_row = []
            for i in range(rows_origin - 1, -1, -1):
                new_row.append(image[i][j])
            new_image.append(new_row)
    elif direction == "L":
        for j in range(columns_origin - 1, -1, -1):
            new_row = []
            for i in range(rows_origin):
                new_row.append(image[i][j])
            new_image.append(new_row)

    return new_image


def mse_formula(n: int, m: int,
                image_pixel: int,
                patch_pixel: int) -> float:
    """Calculate MSE contribution for a single pair of pixels."""
    return (1 / (n * m)) * (image_pixel - patch_pixel) ** 2


def mse_loop(patch_rows: int,
             patch_columns: int,
             image: SingleChannelImage,
             patch: SingleChannelImage,
             t: int,
             s: int) -> float:
    """Compute MSE between patch and image subregion at (t, s)."""
    sum_mse = 0.0
    for i in range(patch_rows):
        for j in range(patch_columns):
            sum_mse += mse_formula(
                patch_rows,
                patch_columns,
                image[t + i][s + j],
                patch[i][j],
            )
    return sum_mse


def find_in_dictionary(dictionary: Dict[Any, Any],
                       item: Any) -> Optional[Tuple[Any, Any]]:
    """Return first (key, value) pair whose value equals item."""
    for key, value in dictionary.items():
        if value == item:
            return key, value
    return None


def get_best_match(image: SingleChannelImage,
                   patch: SingleChannelImage) -> Tuple[Tuple[int, int], float]:
    """Find best patch location in image using MSE."""
    img_rows, img_columns = list_dimensions(image, 2)
    patch_rows, patch_columns = list_dimensions(patch, 2)

    height_difference = img_rows - patch_rows + 1
    width_difference = img_columns - patch_columns + 1

    distances: Dict[Tuple[int, int], float] = {}
    for t in range(height_difference):
        for s in range(width_difference):
            distances[(t, s)] = mse_loop(
                patch_rows, patch_columns, image, patch, t, s
            )

    min_value = min(distances.values())
    # per original logic, rely on find_in_dictionary
    key_value = find_in_dictionary(distances, min_value)
    # `find_in_dictionary` always finds an item here
    return key_value  # type: ignore[return-value]


def loop_resizes(image: SingleChannelImage,
                 num: int) -> Tuple[SingleChannelImage, ...]:
    """Resize image by half, `num` times, collecting all results."""
    list_images = [image]
    for i in range(num):
        current = list_images[i]
        new_h = int(len(current) / 2)
        new_w = int(len(current[0]) / 2)
        list_images.append(resize(current, new_h, new_w))

    list_images.pop(0)
    return tuple(list_images)


def neighbor_pixels(x: int, y: int) -> Tuple[int, int]:
    """Return top-left of 3x3 neighborhood around (x, y)."""
    first_x = 0 if x == 0 else x - 1
    first_y = 0 if y == 0 else y - 1
    return first_x, first_y


def slice_list(img: SingleChannelImage,
               first_x_index: int,
               last_x_index: int,
               first_y_index: int,
               last_y_index: int) -> SingleChannelImage:
    """Return rectangular slice of 2D image."""
    sliced_img = copy.deepcopy(img[first_x_index:last_x_index + 1])
    for i in range(last_x_index + 1 - first_x_index):
        sliced_img[i] = sliced_img[i][first_y_index:last_y_index + 1]
    return sliced_img


def loop_search(dictionary: Dict[int, list],
                num: int,
                *args: SingleChannelImage) -> Dict[int, list]:
    """Multi-scale refinement search for patch location."""
    for i in range(0, 5, 2):
        coarse_loc = dictionary[num][i // 2][0]
        x, y = coarse_loc
        x *= 2
        y *= 2

        start_x, start_y = neighbor_pixels(x, y)

        patch_rows, patch_columns = list_dimensions(args[i], 2)
        base_image = args[i + 1]

        last_x = min(x + patch_rows, len(base_image) - 1)
        last_y = min(y + patch_columns, len(base_image[0]) - 1)

        patch_slice = slice_list(base_image, start_x, last_x, start_y, last_y)

        best_loc, mse = get_best_match(patch_slice, args[i])
        best_x = best_loc[0] + start_x
        best_y = best_loc[1] + start_y

        dictionary[num].append(((best_x, best_y), mse))
    return dictionary


def find_patch_in_img(image: SingleChannelImage,
                      patch: SingleChannelImage) -> Dict[int, list]:
    """Search for patch in image across multiple scales and rotations."""
    image_2, image_4, image_8 = loop_resizes(image, 3)

    final_dictionary: Dict[int, list] = {i: [] for i in range(0, 271, 90)}
    for angle in range(0, 271, 90):
        patch_2, patch_4, patch_8 = loop_resizes(patch, 3)

        final_dictionary[angle].append(get_best_match(image_8, patch_8))
        final_dictionary = loop_search(
            final_dictionary,
            angle,
            patch_4, image_4,
            patch_2, image_2,
            patch, image,
        )

        patch = copy.deepcopy(rotate_90(patch, "L"))
    return final_dictionary


if __name__ == '__main__':
    pass