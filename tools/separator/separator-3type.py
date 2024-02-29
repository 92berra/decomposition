import argparse, glob, io, os, cv2, numpy as np
from jamo import h2j, j2hcj
from PIL import Image, ImageFont, ImageDraw

DEFAULT_IMAGE_DIR = '../../images/target'
DEFAULT_OUTPUT_DIR = '../../images/target-split-3type'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def separate_3type(img_dir, output_dir):
    
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
        #vowel_2 = ['ㅗ','ㅛ','ㅜ','ㅠ','ㅡ']
        vowel_3 = ['ㅘ','ㅙ','ㅚ','ㅝ','ㅞ','ㅟ','ㅢ']
        
        if jamo_dict[j][1] in vowel_3 and len(jamo_dict[j]) == 2:
            # 조합유형 3 일 때만 아래 명령어 수행(이미지 열기, 컨투어 찾기)
            
            # 초성: ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅍ ㅎ
            # 중성: ㅘ ㅙ ㅚ ㅝ ㅞ ㅟ ㅢ
            
            # image 불러와서 findContours
            image = cv2.imread(images_list[j])
            image_copy = image.copy()
            img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            
            center_points_dict = []
            center_points = {}
            sorted_center_points = []
            
            #print(f"{chr(int(char_unicode,16))}, {len(contours)}")
            # len(contours) 갯수만큼 반복
            for i in range(1, len(contours)):

                # contours[i]를 감싼 bbox 생성
                x,y,w,h = cv2.boundingRect(contours[i])
            
                # contours[i] 의 중앙값 계산
                # dictionary 형태로 저장 > key는 contours의 index, value는 contours의 중앙값
                center_x = x + w // 2
                center_y = y + h // 2
                center_points[i] = (center_x + center_y)
                center_points_dict.append((i, center_points[i]))
                
            # x+y값 기준으로 오름차순 정렬
            sorted_center_points = sorted(center_points_dict, key=lambda x: x[1])
                
            # x+y값 기준으로 오름차순 정렬한 인덱스 가져오기
            sorted_contours_indices = [index for index, _ in sorted_center_points]
                
            if jamo_dict[j][0]=='ㅊ':
                
                # 삐침이 붙어있는 경우
                # 1,2,5,11,12,14,15,18,19,21,23,29,32,35,38,48,53,55,58,59,64,65,75,76,78,82,85,92,94,95
                if fontname == '1' or fontname == '2' or fontname == '5' or fontname == '11' or fontname == '12' \
                or fontname == '14' or fontname == '15' or fontname == '18' or fontname == '19' or fontname == '21' \
                or fontname == '23' or fontname == '29' or fontname == '32' or fontname == '35' or fontname == '38' \
                or fontname == '48' or fontname == '53' or fontname == '55' or fontname == '58' or fontname == '59' \
                or fontname == '64' or fontname == '65' or fontname == '75' or fontname == '76' or fontname == '78' \
                or fontname == '82' or fontname == '85' or fontname == '92' or fontname == '94' or fontname == '95':
                    
                    initial_component = contours[sorted_contours_indices[0]]
                            
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
   
                # 획이 3개인 경우
                # 16,33,37,49
                elif fontname == '16' or fontname == '33' or fontname == '37' or fontname == '49':
                    
                    initial_component_1 = contours[sorted_contours_indices[0]]
                    initial_component_2 = contours[sorted_contours_indices[1]]
                    initial_component_3 = contours[sorted_contours_indices[2]]
                            
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_3)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                
                # 초성, 중성이 붙어있는 경우 
                # 20,22,24,67,70,79,87,90,93
                #elif fontname == '22' or fontname == '24' or fontname == '67' or fontname == '70' \
                #or fontname == '79' or fontname == '87' or fontname == '90' or fontname == '93':
                
                
                # 삐침이 분리되어있는 경우
                # 3,4,6,7,8,9,10,13,17,25,26,27,28,30,31,34,36,39,40,41,42,43,44,45,46,47,50,51,52
                # 54,56,57,60,61,62,63,66,68,69,71,72,73,74,77,80,81,83,84,86,88,89,91,96,97,98
                elif fontname == '3' or fontname == '4' or fontname == '6' or fontname == '7' or fontname == '8' \
                or fontname == '9' or fontname == '10' or fontname == '13' or fontname == '17' \
                or fontname == '25' or fontname == '26' or fontname == '27' or fontname == '28' or fontname == '30' \
                or fontname == '31' or fontname == '34' or fontname == '36' or fontname == '39' or fontname == '40' \
                or fontname == '41' or fontname == '42' or fontname == '43' or fontname == '44' or fontname == '45' \
                or fontname == '46' or fontname == '47' or fontname == '50' or fontname == '51' or fontname == '52' \
                or fontname == '54' or fontname == '56' or fontname == '57' or fontname == '60' or fontname == '61' \
                or fontname == '62' or fontname == '63' or fontname == '66' or fontname == '68' or fontname == '69' \
                or fontname == '71' or fontname == '72' or fontname == '73' or fontname == '74' or fontname == '77' \
                or fontname == '80' or fontname == '81' or fontname == '83' or fontname == '84' or fontname == '86' \
                or fontname == '88' or fontname == '89' or fontname =='91' or fontname == '96' or fontname == '97' \
                or fontname == '98':
                    
                    initial_component_1 = contours[sorted_contours_indices[0]]
                    initial_component_2 = contours[sorted_contours_indices[1]]
                            
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                
            elif jamo_dict[j][0]=='ㅇ':
                initial_component_1 = contours[sorted_contours_indices[0]]
                initial_component_2 = contours[sorted_contours_indices[1]]
                initial_component = np.vstack([initial_component_1, initial_component_2])
                        
                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_1.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)
                    
                cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                file_string = f'{fontname}_{char_unicode}_2.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,image_copy)

            elif jamo_dict[j][0]=='ㄸ':
                    
                # ㄷ이 분리되어 있는 경우
                # 1,2,4,5,6,8,9,12,15,19,31,32,37,39,40,41,47,50,51,52,53,61,64,67,72,75,77,79,81,83,84,86,91,92,94
                if fontname == '1' or fontname == '2' or fontname == '4' or fontname == '5' or fontname == '6' or fontname == '8' \
                or fontname == '9' or fontname == '12' or fontname == '19' or fontname == '28' or fontname == '31' or fontname == '32' or fontname == '37' \
                or fontname == '39' or fontname == '40' or fontname == '41' or fontname == '47' or fontname == '50' or fontname == '51' or fontname == '52' \
                or fontname == '53' or fontname == '61' or fontname == '64' or fontname == '67' or fontname == '72' or fontname == '75' or fontname == '77' \
                or fontname == '79' or fontname == '81' or fontname == '83' or fontname == '84' or fontname == '86' or fontname == '91' or fontname == '92' \
                or fontname == '94':
                        
                    initial_component_1 = contours[sorted_contours_indices[0]]
                    initial_component_2 = contours[sorted_contours_indices[1]]
                        
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                                
                    
                # ㄷ이 붙어있는 경우
                # 7,10,11,14,18,20,22,23,25,26,29,30,35,45,48,49,55,56,58,59,60,62,68,69,70,71,73,78,80,82,87,93,97,98
                elif fontname == '7' or fontname == '10' or fontname == '11' or fontname == '14' or fontname == '18' or fontname == '20' \
                or fontname == '22' or fontname == '23' or fontname == '25' or fontname == '26' or fontname == '29' or fontname == '30' or fontname == '35' \
                or fontname == '45' or fontname == '48' or fontname == '49' or fontname == '55' or fontname == '56' or fontname == '58' or fontname == '59' \
                or fontname == '60' or fontname == '62' or fontname == '68' or fontname == '69' or fontname == '70' or fontname == '71' or fontname == '73' \
                or fontname == '78' or fontname == '80' or fontname == '82' or fontname == '87' or fontname == '93' or fontname == '97' or fontname == '98':
                    
                    initial_component_1 = contours[sorted_contours_indices[0]]
                    initial_component_2 = contours[sorted_contours_indices[1]]
                    initial_component = np.vstack([initial_component_1, initial_component_2])
                        
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                        
                # ㄷ이 아래에만 붙어있는 경우
                # 3,13,17,21,24,27,34,36,38,43,44,46,54,57,63,65,66,74,76,85,88,89,90,95,96
                elif fontname == '3' or fontname == '13' or fontname == '15' or fontname == '17' or fontname == '21' or fontname == '24' or fontname == '27' \
                or fontname == '34' or fontname == '36' or fontname == '38' or fontname == '43' or fontname == '44' or fontname == '46' or fontname == '54' \
                or fontname == '57' or fontname == '63' or fontname == '65' or fontname == '66' or fontname == '74' or fontname == '76' or fontname == '85' \
                or fontname == '88' or fontname == '89' or fontname == '90' or fontname == '95' or fontname == '96':
                        
                    initial_component = contours[sorted_contours_indices[0]]
                        
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                # ㄷ이 3개의 컨투어로 잡히는 경우
                # 16,28,33,42,
                elif fontname == '16' or fontname == '33' or fontname == '42':
                    initial_component_1 = contours[sorted_contours_indices[0]]
                    initial_component_2 = contours[sorted_contours_indices[1]]
                    initial_component_3 = contours[sorted_contours_indices[2]]
                        
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_3)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-dir', type=str, dest='img_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR)
   
    args = parser.parse_args()

    separate_3type(args.img_dir, args.output_dir)