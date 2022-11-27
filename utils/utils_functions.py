import random
import cv2
import numpy as np
from scipy.fft import fft2, ifft2
from skimage.io import imshow, show, imread, imsave

from consts import consts


# Fourier Transform related part
def calculate_fft_matrix(image):
    return fft2(image)


def calculate_inverse_fft_matrix(image_complex):
    return ifft2(image_complex)


def calculate_phase_matrix_from_complex_matrix(image):
    return np.angle(image)


def calculate_abs_matrix_from_complex_matrix(image):
    return np.abs(image)


def calculate_complex_matrix_from_abs_and_phase(r, phi):
    func = np.vectorize(calculate_complex_value)
    return func(r, phi)


def calculate_complex_value(r, phi):
    im_phi = complex(0, phi)
    return r * np.exp(im_phi)


# realization of insert formulas
def add_embedding(f, beta, omega, alpha=1):
    return f + alpha * beta * omega


def multiply_embedding(f, omega, alpha=consts.ALPHA, beta=consts.BETA):
    result = f * (1 + alpha * beta * omega)
    return result


# read / save
def save_image(image, path):
    return imsave(path, image)


def read_image(path, as_gray=True):
    return imread(path, as_gray=as_gray)


# getting special zone functions
def get_H_zone(container_abs_matrix):
    modifier = 0.25
    matrix_shape = container_abs_matrix.shape

    shape = (matrix_shape[0] * modifier, matrix_shape[1] * modifier)

    y_border_size = shape[1]
    x_border_size = shape[0]

    left_border = int(x_border_size)
    upper_border = int(y_border_size)

    right_border = int(matrix_shape[0] - x_border_size)
    lower_border = int(matrix_shape[1] - y_border_size)
    slice = container_abs_matrix[left_border:right_border, upper_border:lower_border]
    result = np.copy(slice)
    return result


def merge_pictures_H_zone(image_source, snipped_part):
    modifier = 0.25
    image_shape = image_source.shape
    result_shape = (image_shape[0] * modifier, image_shape[1] * modifier)
    y_border_size = result_shape[1]
    x_border_size = result_shape[0]

    left_border = int(x_border_size)
    upper_border = int(y_border_size)

    right_border = int(image_shape[0] - x_border_size)
    lower_border = int(image_shape[1] - y_border_size)

    result = np.copy(image_source)
    result[left_border:right_border, upper_border:lower_border] = snipped_part

    return result


def split_H_zone_to_4_parts(H_zone):
    x_center, y_center = int(H_zone.shape[0] / 2), int(H_zone.shape[1] / 2)

    left_up = H_zone[0:x_center, 0:y_center]
    right_up = H_zone[0:x_center, y_center:H_zone.shape[1]]
    left_down = H_zone[x_center:H_zone.shape[0], 0:y_center]
    right_down = H_zone[x_center:H_zone.shape[0], y_center:H_zone.shape[1]]
    return [left_up, right_up, left_down, right_down]


def merge_pictures_H_zone_parts(image, snipped_parts):
    # получение нужного размера - возвращется новая матрица
    new_H_zone = get_H_zone(image)

    x_center, y_center = int(new_H_zone.shape[0] / 2), int(new_H_zone.shape[1] / 2)

    new_H_zone[0:x_center, 0:y_center] = snipped_parts[0]
    new_H_zone[0:x_center, y_center:image.shape[1]] = snipped_parts[1]
    new_H_zone[x_center:image.shape[0], 0:y_center] = snipped_parts[2]
    new_H_zone[x_center:image.shape[0], y_center:image.shape[1]] = snipped_parts[3]

    return merge_pictures_H_zone(image, new_H_zone)


# work with WM functions
def calculate_detection_proximity_measure(omega, omega_changed):
    nominator = np.sum(omega * omega_changed)
    #print("nominator")
    #print(nominator)
    denominator = (np.sqrt(np.sum(omega ** 2)) * np.sqrt(np.sum(omega_changed ** 2)))
    #print("denominator")
    #print(denominator)
    if nominator >= 0:
        return nominator / denominator
    else:
        return 0

def calculate_extracted_watermark(f_w, f, alpha=consts.ALPHA, beta=consts.BETA):
    result = (f_w - f) / (beta * alpha * f)
    return result


def generate_watermark_as_pseudo_sequence(length, math_expectation=consts.M, sigma=consts.SIGMA,
                                          key=consts.KEY_VALUE_FOR_SEED):
    random.seed(key)
    result = np.zeros(length)
    for i in range(0, length, 1):
        result[i] = random.gauss(math_expectation, sigma)
    return result, key


def border_processing(value):
    if value > 255:
        return 255
    if value < 0:
        return 0
    return value


def get_container_with_watermark(container, index_zone=0, m=consts.M, sigma=consts.SIGMA, alpha=consts.ALPHA,
                                 beta=consts.BETA,
                                 key=consts.KEY_VALUE_FOR_SEED):
    # 1. Get complex matrix of container
    container_complex_matrix = calculate_fft_matrix(container)

    # 2. Get matrix of abs and phase of container
    abs_fft_container = calculate_abs_matrix_from_complex_matrix(container_complex_matrix)
    phase_fft_container = calculate_phase_matrix_from_complex_matrix(container_complex_matrix)

    # 3. Snipping
    H_zone = get_H_zone(abs_fft_container)
    parts_of_H_zone = split_H_zone_to_4_parts(H_zone)

    # 4. Get watermark length
    watermark_length = int(H_zone.shape[0] * H_zone.shape[1] / 4)

    # 5. Get watermark
    watermark = generate_watermark_as_pseudo_sequence(
        watermark_length, m, sigma, key)[0].reshape(int(H_zone.shape[0] / 2), int(H_zone.shape[1] / 2))

    # 6. Embedding
    parts_of_H_zone[index_zone] = multiply_embedding(parts_of_H_zone[index_zone], watermark, alpha, beta)

    # 7. Merge pictures
    abs_matrix_with_watermark = merge_pictures_H_zone_parts(abs_fft_container, parts_of_H_zone)

    # 8. Recover complex matrix

    complex_matrix_with_watermark = calculate_complex_matrix_from_abs_and_phase(abs_matrix_with_watermark,
                                                                                phase_fft_container)
    container_with_watermark = calculate_inverse_fft_matrix(complex_matrix_with_watermark)

    # result = container_with_watermark
    result = np.round(np.real(calculate_inverse_fft_matrix(complex_matrix_with_watermark)))
    func = np.vectorize(border_processing)
    result = func(result)
    return result


def calculate_rho(container):
    fft_container = calculate_fft_matrix(container)

    # 3. Get abs of image (+ phase)
    abs_fft_container = calculate_abs_matrix_from_complex_matrix(fft_container)

    # 4. Snipping
    H_zone = get_H_zone(abs_fft_container)

    # 5.

    new_shape = [1, H_zone.shape[0] * H_zone.shape[1]]

    result_image = read_image('resource/result.png')
    fft_recover = calculate_fft_matrix(result_image)
    abs_fft_recover = calculate_abs_matrix_from_complex_matrix(fft_recover)
    H_zone_recover = get_H_zone(abs_fft_recover).reshape(new_shape[0], new_shape[1])
    watermark_length = H_zone.shape[0] * H_zone.shape[1]
    watermark = generate_watermark_as_pseudo_sequence(watermark_length, 300, 10, consts.KEY_VALUE_FOR_SEED)[0]
    H_zone = H_zone.reshape(new_shape[0], new_shape[1])
    reshaped_watermark = watermark.reshape(new_shape[0], new_shape[1])

    return calculate_detection_proximity_measure(reshaped_watermark,
                                                 calculate_extracted_watermark(H_zone_recover, H_zone, consts.ALPHA))


def calculate_optimal_parameter(initial_container, target_value=0.9, alpha_max_possible_value=1.0, step=0.01,
                                index_zone=0):
    alpha = 0.01
    alphas = list()
    omega_tilda = list()

    rho_max = 0
    alpha_max = 0

    # 2. Get H zone from initial container
    abs_fft_container = calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(initial_container))
    H_zone = get_H_zone(abs_fft_container)
    watermark_length = int(H_zone.shape[0] * H_zone.shape[1] / 4)
    H_zone_parts = split_H_zone_to_4_parts(H_zone)
    watermark = generate_watermark_as_pseudo_sequence(watermark_length)[0]
    while alpha <= alpha_max_possible_value:

        print(f'step {alpha}')
        # 1. Get H zone from changed picture
        changed_container = get_container_with_watermark(initial_container, index_zone=index_zone, alpha=alpha)
        abs_fft_recover = calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(changed_container))
        H_zone_recover = get_H_zone(abs_fft_recover)
        H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_recover)
        watermark_tilda = np.squeeze(np.reshape(
            calculate_extracted_watermark(H_zone_recover_parts[index_zone], H_zone_parts[index_zone], alpha=alpha),
            (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))

        rho = calculate_detection_proximity_measure(watermark, watermark_tilda)

        if rho > rho_max:
            rho_max = rho
            alpha_max = alpha

        if rho >= target_value:
            alphas.append(alpha)
            omega_tilda.append(watermark_tilda)
        alpha += step

    if len(alphas) == 0:
        return alpha_max

    alpha_min = alphas[0]
    PSNR_min = calculate_psnr(watermark, omega_tilda[0])

    for i in range(1, len(alphas)):
        PSNR = calculate_psnr(watermark, omega_tilda[i])
        if PSNR_min > PSNR:
            PSNR_min = PSNR
            alpha_min = alphas[i]

    return alpha_min


def calculate_psnr(watermark, extracted_watermark):
    return cv2.PSNR(watermark, extracted_watermark)


def calculate_rho_psnr_by_alpha(initial_container, alpha, index_zone=0):
    abs_fft_container = calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(initial_container))
    H_zone = get_H_zone(abs_fft_container)
    watermark_length = int(H_zone.shape[0] * H_zone.shape[1] / 4)
    H_zone_parts = split_H_zone_to_4_parts(H_zone)
    watermark = generate_watermark_as_pseudo_sequence(watermark_length)[0]

    changed_container = get_container_with_watermark(initial_container, index_zone=index_zone, alpha=alpha)
    abs_fft_recover = calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(changed_container))
    H_zone_recover = get_H_zone(abs_fft_recover)
    H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_recover)
    watermark_tilda = np.squeeze(np.reshape(
        calculate_extracted_watermark(H_zone_recover_parts[index_zone], H_zone_parts[index_zone], alpha),
        (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))

    rho = calculate_detection_proximity_measure(watermark, watermark_tilda)
    psnr = calculate_psnr(watermark, watermark_tilda)
    return rho, psnr



