import numpy as np
import scipy.fft
from matplotlib import pyplot as plt
from skimage.io import imshow, show, imread
import utils
from utils.utils_functions import get_container_with_watermark, save_image, calculate_fft_matrix, \
    calculate_abs_matrix_from_complex_matrix, get_H_zone, read_image, generate_watermark_as_pseudo_sequence, \
    calculate_phase_matrix_from_complex_matrix, calculate_extracted_watermark, multiply_embedding, \
    split_H_zone_to_4_parts, \
    merge_pictures_H_zone_parts, calculate_detection_proximity_measure, different_fragments, \
    calculate_inverse_fft_matrix, \
    calculate_complex_matrix_from_abs_and_phase, calculate_optimal_parameter
from consts import consts

if __name__ == '__main__':
    # Получаем пустой контейнер
    container = imread("bridge.tif")
    imshow(container)
    show()
    # Получаем контейнер с водяным знаком
    container_with_wm = get_container_with_watermark(container)

    # Сохраняем контейнер с водяным знаком
    #save_image(container_with_wm, 'files/result.png')

    # Считываем контейнер с водяным знаком
    #recovered_container_with_wm = read_image('files/result.png')

    # Находим значения в сектральной области у исходного контейнера
    H_zone = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(container)))

    # Получаем водяной знак который был встроен путем генерации на основе ключа
    watermark_length = H_zone.shape[0] * H_zone.shape[1]
    watermark = generate_watermark_as_pseudo_sequence(watermark_length)[0]

    # Получаем водяной знак который был извлечен из сохраненного контейнера
    # Находим значения в сектральной области у сохраненного контейнера
    # H_zone_recover = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(recovered_container_with_wm)))
    H_zone_recover = get_H_zone(
        calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(container_with_wm)))
    # Получаем извлекаемый водяной знак по формуле 6.10 и представляем его в виде вектора

    extracted_watermark = np.reshape(calculate_extracted_watermark(H_zone_recover, H_zone, consts.ALPHA),
                                     (1, H_zone.shape[0] * H_zone.shape[1]))

    # Считаем меру близости по формуле 6.11
    proximity_measure = calculate_detection_proximity_measure(watermark, extracted_watermark)
    print(f'Proximity measure: {proximity_measure}')
    #opt = calculate_optimal_parameter(container)
    #print(f'OPT Proximity measure: {opt}')


    # +====================================================================================================
    #
    # fft_container = calculate_fft_matrix(container)
    # abs_fft_container = calculate_abs_matrix_from_complex_matrix(fft_container)
    # phase_fft_container = calculate_phase_matrix_from_complex_matrix(fft_container)
    #
    # H_zone = get_H_zone(abs_fft_container)
    # initial_parts = split_image_to_4_parts(H_zone)
    # watermark_length = initial_parts[0].shape[0] * initial_parts[0].shape[1]
    # watermark = generate_watermark_as_pseudo_sequence(watermark_length, 300, 10, consts.KEY_VALUE_FOR_SEED)[0]
    #
    # for i in range(0, 4, 1):
    #     initial_parts[i] = multiply_embedding(initial_parts[i],
    #                                           consts.BETA, watermark.reshape(initial_parts[i].shape[0],
    #                                                                          initial_parts[i].shape[1]),
    #                                           consts.ALPHA)
    #
    # abs_container_with_watermark = merge_pictures_H_zone_parts(abs_fft_container, initial_parts)
    # complex_container_with_watermark = calculate_complex_matrix_from_abs_and_phase(abs_container_with_watermark,
    #                                                                                phase_fft_container)
    # recovered_container_with_wm = calculate_inverse_fft_matrix(complex_container_with_watermark)
    # save_image(recovered_container_with_wm, 'files/Paul.png')
    #
    # # result_image = read_image('files/result.png')
    # # abs_fft_container = calculate_abs_matrix_from_complex_matrix(calculate_inverse_fft_matrix(container))
    # # H_zone = get_H_zone(abs_fft_container)
    # # initial_parts = split_image_to_4_parts(H_zone)
    #
    # # different_fragments(initial_parts, result_image, watermark)
    #
    # # alpha_result = get_optimal_parameter(container)
    # # print(f'{alpha_result}')
