import argparse, glob, io, os, cv2, numpy as np
from jamo import h2j, j2hcj
from PIL import Image, ImageFont, ImageDraw

DEFAULT_IMAGE_DIR = '../../images/target'
DEFAULT_OUTPUT_DIR = '../../images/target-split-2type'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def separate_2type(img_dir, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir))
        
    images_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
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

        #vowel_1 = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ']
        vowel_2 = ['ㅗ','ㅛ','ㅜ','ㅠ','ㅡ']
        #vowel_3 = ['ㅘ','ㅙ','ㅚ','ㅝ','ㅞ','ㅟ','ㅢ']
        
        # image 불러와서 findContours
        image = cv2.imread(images_list[j])
        image_copy = image.copy()
        img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        
        center_points = {}
        sorted_center_points = []
        
        if jamo_dict[j][1] in vowel_2 and len(jamo_dict[j]) == 2:
            # 초성: ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅍ ㅎ
            # 중성: ㅗ ㅛ ㅜ ㅠ ㅡ
            
            # len(contours) 갯수만큼 반복
            for i in range(1, len(contours)):

                # contours[i]를 감싼 bbox 생성
                x,y,w,h = cv2.boundingRect(contours[i])

                # contours[i] 의 중앙값 계산
                # dictionary 형태로 저장 > key는 i, value는 contours의 중앙값
                center_x = x + w // 2
                center_y = y + h // 2
                center_points[i] = (center_x, center_y)
                sorted_center_points.append((i, center_points[i]))
            
            # y값 기준으로 오름차순 정렬
            sorted_center_points = sorted(sorted_center_points, key=lambda x: x[1][1])
        
            # y값 기준으로 오름차순 정렬한 인덱스 가져오기
            sorted_contours_indices = [index for index, _ in sorted_center_points]
            
            middle_component = contours[sorted_contours_indices[-1]]
        
            mask = np.zeros((256,256,3), np.uint8) + 255
            cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
            file_string = f'{fontname}_{char_unicode}_2.png'
            file_path = os.path.join(output_dir, file_string)
            cv2.imwrite(file_path,mask)
        
            cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
            file_string = f'{fontname}_{char_unicode}_1.png'
            file_path = os.path.join(output_dir, file_string)
            cv2.imwrite(file_path,image_copy)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-dir', type=str, dest='img_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR)
   
    args = parser.parse_args()

    separate_2type(args.img_dir, args.output_dir)