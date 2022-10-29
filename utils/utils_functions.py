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


def multiply_embedding(f, omega,  alpha=consts.ALPHA, beta=consts.BETA):
    return f * (1 + alpha * beta * omega)


# read / save
def save_image(image, path):
    return imsave(path, image)


def read_image(path, as_gray=True):
    return imread(path, as_gray=as_gray)


# getting special zone functions
def get_H_zone(image, modifier=0.25):
    image_shape = image.shape
    result_shape = (image_shape[0] * modifier, image_shape[1] * modifier)
    y_border_size = result_shape[1]
    x_border_size = result_shape[0]

    left_border = int(x_border_size - 1)
    upper_border = int(y_border_size - 1)

    right_border = int(image_shape[0] - x_border_size - 1)
    lower_border = int(image_shape[1] - y_border_size - 1)

    result = np.copy(image[left_border:right_border, upper_border:lower_border])
    return result


def split_H_zone_to_4_parts(image):
    x_center, y_center = int(image.shape[0] / 2), int(image.shape[1] / 2)

    left_up = image[0:x_center, 0:y_center]
    right_up = image[0:x_center, y_center:image.shape[1]]
    left_down = image[x_center:image.shape[0], 0:y_center]
    right_down = image[x_center:image.shape[0], y_center:image.shape[1]]
    return [left_up, right_up, left_down, right_down]


def merge_pictures_H_zone(image_source, snipped_part, modifier=0.25):
    image_shape = image_source.shape
    result_shape = (image_shape[0] * modifier, image_shape[1] * modifier)
    y_border_size = result_shape[1]
    x_border_size = result_shape[0]

    left_border = int(x_border_size - 1)
    upper_border = int(y_border_size - 1)

    right_border = int(image_shape[0] - x_border_size - 1)
    lower_border = int(image_shape[1] - y_border_size - 1)

    result = np.copy(image_source)
    result[left_border:right_border, upper_border:lower_border] = snipped_part

    return result


def merge_pictures_H_zone_parts(image, snipped_parts):
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
    denominator = (np.sqrt(np.sum(omega ** 2)) * np.sqrt(np.sum(omega_changed ** 2)))
    return nominator / denominator


def calculate_extracted_watermark(f_w, f, alpha=consts.ALPHA, beta=consts.BETA):
    return (f_w - f) / (beta * alpha * f)


def generate_watermark_as_pseudo_sequence(length, math_expectation=consts.M, sigma=consts.SIGMA,
                                          key=consts.KEY_VALUE_FOR_SEED):
    random.seed(key)
    result = np.zeros(length)
    for i in range(0, length, 1):
        result[i] = random.gauss(math_expectation, sigma)
    return result, key


def get_container_with_watermark(container, m=consts.M, sigma=consts.SIGMA, alpha=consts.ALPHA, beta=consts.BETA, key=consts.KEY_VALUE_FOR_SEED):
    # 1. Get complex matrix of container
    container_complex_matrix = calculate_fft_matrix(container)

    # 2. Get matrix of abs and phase of container
    abs_fft_container = calculate_abs_matrix_from_complex_matrix(container_complex_matrix)
    phase_fft_container = calculate_phase_matrix_from_complex_matrix(container_complex_matrix)

    # 3. Snipping
    H_zone = get_H_zone(abs_fft_container)
    parts_of_H_zone = split_H_zone_to_4_parts(H_zone)
    first_quarter = parts_of_H_zone[0]

    # 4. Get watermark length
    watermark_length = int(H_zone.shape[0] * H_zone.shape[1] / 4)

    # 5. Get watermark
    watermark = generate_watermark_as_pseudo_sequence(
        watermark_length, m, sigma, key)[0].reshape(int(H_zone.shape[0]/2), int(H_zone.shape[1]/2))

    # 6. Embedding
    parts_of_H_zone[0] = multiply_embedding(first_quarter, beta, watermark, alpha)

    # 7. Merge pictures

    abs_container_with_watermark = merge_pictures_H_zone_parts(abs_fft_container, parts_of_H_zone)

    # 8. Recover complex matrix
    complex_container_with_watermark = calculate_complex_matrix_from_abs_and_phase(abs_container_with_watermark,
                                                                                   phase_fft_container)

    # return np.round(np.real(calculate_inverse_fft_matrix(complex_container_with_watermark))).astype(np.uint8)
    return calculate_inverse_fft_matrix(complex_container_with_watermark)


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


def calculate_optimal_parameter(initial_container, target_value=0.9, max_possible_value=1.0):
    alpha = 0.001
    alphas = list()
    omega = list()
    omega_tilda = list()

    rho_max = 0
    alpha_max = 0

    while alpha < max_possible_value:

        print(f'step {alpha}')
        # 1. Get H zone from changed picture
        changed_container = get_container_with_watermark(initial_container, consts.M,
                                                         consts.SIGMA, alpha, consts.BETA, consts.KEY_VALUE_FOR_SEED)
        abs_fft_recover = calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(changed_container))
        H_zone_recover = get_H_zone(abs_fft_recover)

        # 2. Get H zone from initial picture
        abs_fft_container = calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(initial_container))
        H_zone = get_H_zone(abs_fft_container)

        new_shape = [1, H_zone_recover.shape[0] * H_zone_recover.shape[1]]
        watermark = generate_watermark_as_pseudo_sequence(new_shape[1],
                                                          consts.M, consts.SIGMA, consts.KEY_VALUE_FOR_SEED)[0]
        watermark_tilda = calculate_extracted_watermark(
            H_zone_recover.reshape(new_shape[0], new_shape[1]),
            H_zone.reshape(new_shape[0], new_shape[1]),
            consts.ALPHA)
        rho = calculate_detection_proximity_measure(watermark.reshape(new_shape[0], new_shape[1]), watermark_tilda)

        if rho > rho_max:
            rho_max = rho
            alpha_max = alpha

        if rho >= target_value:
            alphas.append(alpha)
            omega.append(watermark)
            omega_tilda.append(watermark_tilda)
        alpha += 0.001

    if len(alphas) == 0:
        return alpha_max

    alpha_min = alphas[0]
    PSNR_min = cv2.PSNR(omega[0], omega_tilda[0])

    for i in range(1, len(alphas)):
        PSNR = cv2.PSNR(omega[i], omega_tilda[i])
        if PSNR_min > PSNR:
            PSNR_min = PSNR
            alpha_min = alphas[i]

    return alpha_min


def different_fragments(parts, result_image, watermark):
    fft_recover = calculate_fft_matrix(result_image)
    abs_fft_recover = calculate_abs_matrix_from_complex_matrix(fft_recover)
    H_zone_recover = get_H_zone(abs_fft_recover)
    recover_parts = split_H_zone_to_4_parts(H_zone_recover)

    new_shape = [1, int(H_zone_recover.shape[0] / 2) * int(H_zone_recover.shape[1] / 2)]

    for i in range(0, 4, 1):
        watermark_tilda = calculate_extracted_watermark(recover_parts[i].reshape(new_shape[0], new_shape[1]),
                                                        parts[i].reshape(new_shape[0], new_shape[1]),
                                                        consts.ALPHA)
        watermark = watermark.reshape(new_shape[0], new_shape[1])
        watermark_tilda = watermark_tilda.reshape(new_shape[0], new_shape[1])
        rho = calculate_detection_proximity_measure(watermark,
                                                    watermark_tilda)
        psnr = cv2.PSNR(watermark, watermark_tilda)
        print(f'Result {i}: Rho={rho}; PSNR={psnr}')