import random
import time

import numpy as np
import math

import scipy
from PIL import Image
from matplotlib import pyplot as plt
from skimage.io import imshow, show
from skimage.transform import rescale

from consts import consts
from utils.utils_functions import calculate_detection_proximity_measure, get_H_zone, split_H_zone_to_4_parts, \
    calculate_abs_matrix_from_complex_matrix, calculate_fft_matrix, calculate_extracted_watermark

VALUE_OF_ONE = 1
MAX_BRIGHTNESS_VALUE = int(255)
MIN_BRIGHTNESS_VALUE = int(0)
SEED = 42
random.seed(SEED)
# TASK WNoise requirement
M = 0


# 14 Cut Scale GaussBlur WhNoise 2 искажения
# № искажения Параметр 𝑝 𝑝𝑚𝑖𝑛 𝑝𝑚𝑎𝑥 Δ𝑝

# Cut Доля площади 𝜗 0.2 0.9 0.1        + относительно устойчив

# Scale К-т масштабирования 0.55 1.45 0.15      - неустойчив

# GaussBlur Параметр размытия 𝜎 1 4 0.5     - неустойчив

# WhNoise Дисперсия шума Dξ 400 1000 100    +- мера близости 50 - 70 процентов

# «2 искажения» - по результатам исследования стойкости СВИ
# выбрать два искажения из числа рассмотренных, по отношению к которым система обладает наибольшей стойкостью.
# Далее применить последовательно к носителю информации одно и другое искажение, используя все сочетания параметров,
# и составить таблицу, характеризующую стойкость СВИ к комбинации двух выбранных искажений.
# Сделать вывод по результатам эксперимента.


def apply_cut_distortion(C_w, C, teta=0.5):
    C_w_shape = np.shape(C_w)
    result = np.copy(C_w)
    n1 = int(C_w_shape[0] * np.sqrt(teta))
    n2 = int(C_w_shape[1] * np.sqrt(teta))
    for i in range(n1 + VALUE_OF_ONE, C_w_shape[0], 1):
        for j in range(n2 + VALUE_OF_ONE, C_w_shape[1], 1):
            result[i, j] = C[i, j]

    return result


def apply_cut_distortion_alter(C_w, C, teta=0.5):
    C_w_shape = np.shape(C_w)
    result = np.copy(C_w)
    n1 = int(C_w_shape[0] * np.sqrt(teta))
    n2 = int(C_w_shape[1] * np.sqrt(teta))

    for i in range(0, n1 + VALUE_OF_ONE, 1):
        for j in range(0, n2 + VALUE_OF_ONE, 1):
            result[i, j] = C[i, j]

    return result


def apply_cut_distortion_test(C_w, C, teta=0.5):
    C_w_shape = np.shape(C)
    result = np.copy(C_w)
    for i in range(0, C_w_shape[0], 1):
        for j in range(0, C_w_shape[1], 1):
            result[i, j] = 0

    return result


def analyze_cut_distortion(C, C_w, W, param_start=0.2, param_stop=0.9, step=0.1, cut_dist_func=apply_cut_distortion):
    list_of_proximity_measures = []
    list_of_params_value = []
    # list_of_cuts = []
    H_zone = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(C)))
    H_zone_parts = split_H_zone_to_4_parts(H_zone)

    parameter = param_start
    while parameter <= param_stop:
        tmp_dist_cut = cut_dist_func(C_w, C, parameter)
        # list_of_cuts.append(tmp_dist_cut)
        list_of_params_value.append(parameter)

        H_zone_recover = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(tmp_dist_cut)))
        H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_recover)

        tmp_extracted_watermark_cut = np.squeeze(np.reshape(
            calculate_extracted_watermark(H_zone_recover_parts[0], H_zone_parts[0], consts.ALPHA),
            (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))

        # copy = np.copy(tmp_extracted_watermark_cut)
        # count = np.count_nonzero(np.where(copy < 0, copy, 0))
        # print("count negative for teta = " + str(parameter) + " equals:  " + str(count))

        tmp_prox_measure = calculate_detection_proximity_measure(W, tmp_extracted_watermark_cut)
        list_of_proximity_measures.append(tmp_prox_measure)
        print(f'CUT Proximity measure for teta = ' + str(parameter) + ' ', {tmp_prox_measure})

        tmp_figure = plt.figure()
        tmp_figure.suptitle("Source container with WM and cut result for teta =" + str(parameter))
        sub_1 = tmp_figure.add_subplot(1, 2, 1)
        sub_1.set(title='Source container with watermark')
        imshow(C_w)

        sub_2 = tmp_figure.add_subplot(1, 2, 2)
        sub_2.set(title='Cut distortion')
        imshow(tmp_dist_cut)
        show()
        parameter += step

    figure = plt.figure()
    tmp_figure.suptitle("Proximity measure value dependency from teta")
    plt.plot(list_of_params_value, list_of_proximity_measures, '-k')
    show()


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

def apply_scale(C_w, K):
    C_w_shape = np.shape(C_w)
    scaled_image = rescale(C_w, K)
    scaled_shape = scaled_image.shape
    if K < 1:
        result = np.zeros(C_w.shape)
        result[C_w_shape[0] // 2 - scaled_shape[0] // 2: C_w_shape[0] // 2 + scaled_shape[0] // 2,
        C_w_shape[1] // 2 - scaled_shape[1] // 2: C_w_shape[1] // 2 + scaled_shape[1] // 2] = \
            scaled_image[scaled_shape[0] // 2 - scaled_shape[0] // 2: scaled_shape[0] // 2 + scaled_shape[0] // 2,
            scaled_shape[1] // 2 - scaled_shape[1] // 2: scaled_shape[1] // 2 + scaled_shape[1] // 2]
        return result
    elif K > 1:
        return scaled_image[scaled_shape[0] // 2 - C_w_shape[0] // 2: scaled_shape[0] // 2 + C_w_shape[0] // 2,
               scaled_shape[1] // 2 - C_w_shape[1] // 2: scaled_shape[1] // 2 + C_w_shape[1] // 2]
    else:
        return C_w


def analyze_scale(C, C_w, W, param_start=0.55, param_stop=1.45, step=0.15):
    list_of_proximity_measures = []
    list_of_params_value = []
    H_zone = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(C)))
    H_zone_parts = split_H_zone_to_4_parts(H_zone)
    parameter = param_start
    while parameter <= param_stop:
        tmp_dist_scale = apply_scale(C_w, parameter)
        # list_of_cuts.append(tmp_dist_scale)
        list_of_params_value.append(parameter)

        H_zone_recover = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(tmp_dist_scale)))
        H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_recover)

        tmp_extracted_watermark_cut = np.squeeze(np.reshape(
            calculate_extracted_watermark(H_zone_recover_parts[0], H_zone_parts[0], consts.ALPHA),
            (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))

        tmp_prox_measure = calculate_detection_proximity_measure(W, tmp_extracted_watermark_cut)
        list_of_proximity_measures.append(tmp_prox_measure)
        print(f'SCALE Proximity measure for K = ' + str(parameter) + ' ', {tmp_prox_measure})

        tmp_figure = plt.figure()
        tmp_figure.suptitle("Source container with WM and SCALE result for K =" + str(parameter))
        sub_1 = tmp_figure.add_subplot(1, 2, 1)
        sub_1.set(title='Source container with watermark')
        imshow(C_w)

        sub_2 = tmp_figure.add_subplot(1, 2, 2)
        sub_2.set(title='SCALE distortion')
        imshow(tmp_dist_scale)
        show()
        parameter += step

    figure = plt.figure()
    tmp_figure.suptitle("Proximity measure value dependency from K")
    plt.plot(list_of_params_value, list_of_proximity_measures, '-k')
    show()


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

def exp_value_for_K(m1, m2, M, sigma):
    return math.exp(-(np.power((m1 - M / 2), 2) + np.power((m2 - M / 2), 2))) / (2 * np.power(sigma, 2))


def find_K_coefficient(M, sigma):
    denominator = 0
    for m1 in range(0, M, 1):
        for m2 in range(0, M, 1):
            denominator += exp_value_for_K(m1, m2, M, sigma)
    return 1 / denominator


def get_GaussBlur_pixel_value(n1, n2, M, K, Cw, sigma):
    result = 0
    for m1 in range(0, M, 1):
        for m2 in range(0, M, 1):
            tmp_g = K * exp_value_for_K(m1, m2, M, sigma)
            result += Cw[n1 - m1, n2 - m2] * tmp_g
    return result


def apply_GaussBlur(Cw, sigma=1):

    M = 2 * math.floor(3 * sigma) + 1
    K = find_K_coefficient(M, sigma)
    G = np.zeros((M, M))
    for m1 in range(0, M, 1):
        for m2 in range(0, M, 1):
            G[m1, m2] = K * exp_value_for_K(m1, m2, M, sigma)

    result = scipy.signal.convolve2d(G, Cw)
    # for n1 in range(0, Cw_shape[0], 1):
    #     start = time.time()
    #     for n2 in range(0, Cw_shape[1], 1):
    #         result[n1, n2] = get_GaussBlur_pixel_value(n1, n2, M, K, Cw, sigma)
    #
    #     end = time.time()
    #     print("The time of execution of above program is :",
    #           (end - start) * 10 ** 3, "ms")

    for i in range(0, int(M/2), 1):
        result = np.delete(result, i, 0)
        result = np.delete(result, -i, 0)
        result = np.delete(result, i, 1)
        result = np.delete(result, -i, 1)
    return result


def analyze_GaussBlur(C, C_w, W, param_start=1.0, param_stop=4.0, step=0.5):
    list_of_proximity_measures = []
    list_of_params_value = []
    H_zone = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(C)))
    H_zone_parts = split_H_zone_to_4_parts(H_zone)
    parameter = param_start
    while parameter <= param_stop:
        tmp_dist = apply_GaussBlur(C_w, parameter)

        list_of_params_value.append(parameter)

        H_zone_recover = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(tmp_dist)))
        H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_recover)

        tmp_extracted_watermark_cut = np.squeeze(np.reshape(
            calculate_extracted_watermark(H_zone_recover_parts[0], H_zone_parts[0], consts.ALPHA),
            (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))

        tmp_prox_measure = calculate_detection_proximity_measure(W, tmp_extracted_watermark_cut)
        list_of_proximity_measures.append(tmp_prox_measure)
        print(f'GaussBlur Proximity measure for SD = ' + str(parameter) + ' ', {tmp_prox_measure})

        tmp_figure = plt.figure()
        tmp_figure.suptitle("Source container with WM and GaussBlur result for SD =" + str(parameter))
        sub_1 = tmp_figure.add_subplot(1, 2, 1)
        sub_1.set(title='Source container with watermark')
        imshow(C_w)

        sub_2 = tmp_figure.add_subplot(1, 2, 2)
        sub_2.set(title='GaussBlur distortion')
        imshow(tmp_dist, cmap='gray', vmin=0, vmax=255 )
        show()
        parameter += step

    figure = plt.figure()
    tmp_figure.suptitle("Proximity measure value dependency from SD")
    plt.plot(list_of_params_value, list_of_proximity_measures, '-k')
    show()


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

def add_white_noise_to_pixel(source_value, sigma, apply_border_processing=True):
    gauss_part = random.gauss(M, sigma)
    if apply_border_processing:
        if source_value + gauss_part > MAX_BRIGHTNESS_VALUE:
            return MAX_BRIGHTNESS_VALUE
        else:
            if source_value + gauss_part < 0:
                return MIN_BRIGHTNESS_VALUE
    return source_value + gauss_part


def apply_white_noise(C_w, sigma):
    shape_C_w = np.shape(C_w)
    result = np.copy(C_w)
    for i in range(0, shape_C_w[0], 1):
        for j in range(0, shape_C_w[1], 1):
            result[i, j] = add_white_noise_to_pixel(C_w[i, j], sigma)
    return result


def analyze_white_noise(C, C_w, W, param_start=400, param_stop=1000, step=100):
    list_of_proximity_measures = []
    list_of_params_value = []
    H_zone = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(C)))
    H_zone_parts = split_H_zone_to_4_parts(H_zone)
    parameter = param_start
    while parameter <= param_stop:
        tmp_dist = apply_white_noise(C_w, np.sqrt(parameter))
        # list_of_cuts.append(tmp_dist)
        list_of_params_value.append(parameter)

        H_zone_recover = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(tmp_dist)))
        H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_recover)

        tmp_extracted_watermark_cut = np.squeeze(np.reshape(
            calculate_extracted_watermark(H_zone_recover_parts[0], H_zone_parts[0], consts.ALPHA),
            (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))

        tmp_prox_measure = calculate_detection_proximity_measure(W, tmp_extracted_watermark_cut)
        list_of_proximity_measures.append(tmp_prox_measure)
        print(f'WHITE_NOISE Proximity measure for dispersion = ' + str(parameter) + ' ', {tmp_prox_measure})

        tmp_figure = plt.figure()
        tmp_figure.suptitle("Source container with WM and WHITE_NOISE result for dispersion =" + str(parameter))
        sub_1 = tmp_figure.add_subplot(1, 2, 1)
        sub_1.set(title='Source container with watermark')
        imshow(C_w)

        sub_2 = tmp_figure.add_subplot(1, 2, 2)
        sub_2.set(title='WHITE_NOISE distortion')
        imshow(tmp_dist)
        show()
        parameter += step

    figure = plt.figure()
    tmp_figure.suptitle("Proximity measure value dependency from dispersion")
    plt.plot(list_of_params_value, list_of_proximity_measures, '-k')
    show()


# =====================================================================================================
# =====================================================================================================
# =====================================================================================================

def apply_2_distorsions(C_w,
                        dist_func_1=apply_cut_distortion, parameter1=1,
                        dist_func_2=apply_white_noise, parameter2=1):
    result = dist_func_1(C_w, parameter1)
    return dist_func_2(result, parameter2)


def analyze_2_distortions(C_w, C, W,
                          param1_start=0.2, param1_stop=0.9, step1=0.1,
                          param2_start=400, param2_stop=1000, step2=100,
                          skip_images=False):

    number_of_x_coords = round(((param1_stop - param1_start)/step1)) + VALUE_OF_ONE
    number_of_y_coords = round(((param2_stop - param2_start)/step2)) + VALUE_OF_ONE
    result_data = np.zeros((number_of_x_coords, number_of_y_coords))

    H_zone = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(C)))
    H_zone_parts = split_H_zone_to_4_parts(H_zone)

    i = 0
    parameter1 = param1_start
    while parameter1 <= param1_stop:
        j = 0
        parameter2 = param2_start
        while parameter2 <= param2_stop:
            tmp_dist1 = apply_cut_distortion(C_w, C, parameter1)
            tmp_dist2 = apply_white_noise(tmp_dist1, np.sqrt(parameter2))


            H_zone_recover = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(tmp_dist2)))
            H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_recover)

            tmp_extracted_watermark_cut = np.squeeze(np.reshape(
                calculate_extracted_watermark(H_zone_recover_parts[0], H_zone_parts[0], consts.ALPHA),
                (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))

            tmp_prox_measure = calculate_detection_proximity_measure(W, tmp_extracted_watermark_cut)
            result_data[i, j] = tmp_prox_measure


            print(f'CUT and WHITE_NOISE Proximity measure for teta = ' + str(parameter1) + ' and dispersion = ' +
                  str(parameter2), {tmp_prox_measure})


            if not skip_images:
                tmp_figure = plt.figure(figsize=(10, 10))
                tmp_figure.suptitle("Source container with WM and CUT" + " CUT teta" + str(parameter1) +
                                    " and WHITE_NOISE dispersion =" + str(parameter2))
                sub_1 = tmp_figure.add_subplot(1, 2, 1)
                sub_1.set(title='Source container with watermark')
                imshow(C_w)

                sub_2 = tmp_figure.add_subplot(1, 2, 2)
                sub_2.set(title="CUT teta = " + str(parameter1) + " and WHITE_NOISE dispersion = " + str(parameter2))
                imshow(tmp_dist2)
                show()

            j += 1
            parameter2 += step2

        i += 1
        parameter1 += step1
        print("New teta iteration :)")


    print_result(result_data)
    img_data = (result_data*255).astype(np.uint8)
    img = Image.fromarray(img_data)

    last_figure = plt.figure()
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img)
    last_figure.suptitle("Proximity measure value dependency from dispersion")
    show()

    return result_data


def print_result(result_data):
    print("teta_value from 0.2 to 0.9 with  step = 0.1  - strings")
    print("Dispersion_value from 400 to 1000 with step = 100  - columns")
    string = ''
    for i in range(0, result_data.shape[0], 1):
        for j in range(0, result_data.shape[1], 1):
            string += str(result_data[i, j]) + '\t'
        string += '\n'
    print(string)
    return
