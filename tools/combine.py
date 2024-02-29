import argparse
import glob
import os
import cv2
import numpy as np
from jamo import h2j, j2hcj

# Default data paths.
DEFAULT_INPUT_DIR = '../images/test-split-total'
DEFAULT_OUTPUT_DIR = '../images/test-combine-50-98-empty'

def img_combine(input_dir, output_dir):
    
    images_list = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir))
    
    jamo_dict = {}

    for j in range(len(images_list)):
        
        filename = os.path.basename(images_list[j])
        filename = os.path.splitext(filename)[0]
        split_filename = filename.split('_')
        
        # 파일명에서 필요한 부분 가져오기
        fontname = split_filename[0]
        char_unicode = split_filename[1]

        # 파일명의 unicode 를 한글 문자로 형변환 후 자모분리
        sylla = chr(int(char_unicode, 16))
        jamo = j2hcj(h2j(sylla))
        jamo_list = list(jamo)
        jamo_dict[j] = jamo_list

        #img1_path = os.path.join(input_dir, f'0_{char_unicode}.png')
        #img2_path = os.path.join(input_dir, f'{fontname}_{char_unicode}.png')
        img3_path = os.path.join(input_dir, f'{fontname}_{char_unicode}_1.png')
        img4_path = os.path.join(input_dir, f'{fontname}_{char_unicode}_2.png')
        img5_path = os.path.join(input_dir, f'{fontname}_{char_unicode}_3.png')

        #img1 = cv2.imread(img1_path)
        #img2 = cv2.imread(img2_path)
        img1 = np.zeros((256,256,3), np.uint8) + 255
        img2 = np.zeros((256,256,3), np.uint8) + 255
        img3 = cv2.imread(img3_path)
        img4 = cv2.imread(img4_path)

        if len(jamo_dict[j]) == 2:
            img5 = np.zeros((256,256,3), np.uint8) + 255
        else:
            img5 = cv2.imread(img5_path)

        if img1 is None \
        or img2 is None \
        or img3 is None \
        or img4 is None \
        or img5 is None:
            continue

        hconcat_img = cv2.hconcat([img1, img2, img3, img4, img5])

        output_filename = os.path.join(output_dir, f'{fontname}_{char_unicode}.png')
        cv2.imwrite(output_filename, hconcat_img)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, dest='input_dir', default=DEFAULT_INPUT_DIR, help='')
    parser.add_argument('--output-dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR, help='')
    args = parser.parse_args()

    img_combine(args.input_dir, args.output_dir)
