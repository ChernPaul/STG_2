import time
import numpy as np
import scipy.fft
from matplotlib import pyplot as plt
from skimage.io import imshow, show, imread
import utils
from lab_3_functions.lab_3_functions import apply_cut_distortion, analyze_cut_distortion, apply_cut_distortion_test, \
    apply_cut_distortion_alter, apply_GaussBlur, analyze_scale, analyze_white_noise, analyze_2_distortions, \
    print_result, analyze_GaussBlur
from utils.utils_functions import get_container_with_watermark, save_image, calculate_fft_matrix, \
    calculate_abs_matrix_from_complex_matrix, get_H_zone, read_image, generate_watermark_as_pseudo_sequence, \
    calculate_phase_matrix_from_complex_matrix, calculate_extracted_watermark, multiply_embedding, \
    split_H_zone_to_4_parts, \
    merge_pictures_H_zone_parts, calculate_detection_proximity_measure, \
    calculate_inverse_fft_matrix, \
    calculate_complex_matrix_from_abs_and_phase, calculate_optimal_parameter, merge_pictures_H_zone, \
    calculate_rho_psnr_by_alpha
from consts import consts

if __name__ == '__main__':
    # Получаем пустой контейнер
    container = imread("barb.tif")
    imshow(container)
    show()
    # Получаем контейнер с водяным знаком
    container_with_wm = get_container_with_watermark(container)

    # Сохраняем контейнер с водяным знаком
    save_image(container_with_wm, 'files/result.png')

    # Считываем контейнер с водяным знаком
    rec_container_with_wm = read_image('files/result.png')
    # Показываем контейнер с водяным знаком
    imshow(rec_container_with_wm)
    show()

    """
    # Находим значения в сектральной области у исходного контейнера
    H_zone = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(container)))

    # Получаем водяной знак который был встроен путем генерации на основе ключа
    watermark_length = int(H_zone.shape[0] * H_zone.shape[1] / 4)
    watermark = generate_watermark_as_pseudo_sequence(watermark_length)[0]

    # Получаем водяной знак который был извлечен из сохраненного контейнера
    # Находим значения в сектральной области у сохраненного контейнера
    # H_zone_recover = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(recovered_container_with_wm)))
    H_zone_recover = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(recovered_container_with_wm)))
    # Получаем извлекаемый водяной знак по формуле 6.10 и представляем его в виде вектора
    H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_recover)
    H_zone_parts = split_H_zone_to_4_parts(H_zone)

    extracted_watermark = np.squeeze(np.reshape(
        calculate_extracted_watermark(H_zone_recover_parts[0], H_zone_parts[0], consts.ALPHA),
        (1, int(H_zone.shape[0] * H_zone.shape[1]/4))))

    # Считаем меру близости по формуле 6.11
    proximity_measure = calculate_detection_proximity_measure(watermark, extracted_watermark)
    print(f'Proximity measure: {proximity_measure}')
    opt = calculate_optimal_parameter(container, alpha_max_possible_value=1, step=0.1)
    print(f'OPT alpha : {opt}')
    rho_opt, psnr_opt = calculate_rho_psnr_by_alpha(container, opt)
    print(f'OPT rho and psnr : {rho_opt}, {psnr_opt}')

    rho_some, psnr_some = calculate_rho_psnr_by_alpha(container, 1)
    print(f'Some rho and psnr : {rho_some}, {psnr_some}')

    print("different fragments")
    ind0_rho, ind0_psnr = calculate_rho_psnr_by_alpha(container, 1, 0)
    print(f'Index 0 rho and psnr : {ind0_rho}, {ind0_psnr}')
    ind1_rho, ind1_psnr = calculate_rho_psnr_by_alpha(container, 1, 1)
    print(f'Index 1 rho and psnr : {ind1_rho}, {ind1_psnr}')
    ind2_rho, ind2_psnr = calculate_rho_psnr_by_alpha(container, 1, 2)
    print(f'Index 2 rho and psnr : {ind2_rho}, {ind2_psnr}')
    ind3_rho, ind3_psnr = calculate_rho_psnr_by_alpha(container, 1, 3)
    print(f'Index 3 rho and psnr : {ind3_rho}, {ind3_psnr}')
    """

    H_zone = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(container)))
    watermark_length = int(H_zone.shape[0] * H_zone.shape[1] / 4)
    watermark = generate_watermark_as_pseudo_sequence(watermark_length)[0]

    H_zone_rec = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(rec_container_with_wm)))
    H_zone_recover_parts = split_H_zone_to_4_parts(H_zone_rec)
    H_zone_parts = split_H_zone_to_4_parts(H_zone)
    extracted_watermark = np.squeeze(np.reshape(
        calculate_extracted_watermark(H_zone_recover_parts[0], H_zone_parts[0], consts.ALPHA),
        (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))


    proximity_measure = calculate_detection_proximity_measure(watermark, extracted_watermark)
    print(f'Proximity measure without distortion: {proximity_measure}')

    print('CUT analyze started')
    analyze_cut_distortion(container, rec_container_with_wm, watermark, cut_dist_func=apply_cut_distortion)
    print('CUT analyze ended \n')

    print('SCALE analyze started')
    analyze_scale(container, rec_container_with_wm, watermark)
    print('SCALE analyze ended \n')

    print('WHITE NOISE analyze started')
    analyze_white_noise(container, rec_container_with_wm, watermark)
    print('WHITE NOISE analyze ended \n')


    # uncomment if needed
    """
    print('GaussBlur analyze started')


    start = time.time()
    GaussBlur = apply_GaussBlur(rec_container_with_wm)
    end = time.time()
    print("The time of execution of above program is :",
          (end - start) * 10 ** 3, "ms")


    H_zone_rec_GB = get_H_zone(calculate_abs_matrix_from_complex_matrix(calculate_fft_matrix(GaussBlur)))
    H_zone_recover_parts_GB = split_H_zone_to_4_parts(H_zone_rec_GB)
    H_zone_parts = split_H_zone_to_4_parts(H_zone)
    extracted_watermark_GB = np.squeeze(np.reshape(
        calculate_extracted_watermark(H_zone_recover_parts_GB[0], H_zone_parts[0], consts.ALPHA),
        (1, int(H_zone.shape[0] * H_zone.shape[1] / 4))))

    proximity_measure = calculate_detection_proximity_measure(watermark, extracted_watermark_GB)
    print(f'Proximity measure GB: {proximity_measure}')
    figure_0 = plt.figure()
    figure_0.suptitle("Source container and GaussBlur result")
    sub_1 = figure_0.add_subplot(1, 2, 1)
    sub_1.set(title='Container with wm')
    imshow(rec_container_with_wm)

    sub_2 = figure_0.add_subplot(1, 2, 2)
    sub_2.set(title='GaussBlur')
    imshow(GaussBlur, cmap='gray', vmin=0, vmax=255)
    show()


    analyze_GaussBlur(container, rec_container_with_wm, watermark)
    print('GaussBlur analyze ended')

    """
    # set True to skip 56 images
    table = analyze_2_distortions(rec_container_with_wm, container, watermark, skip_images=True)


