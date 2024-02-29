import argparse, glob, io, os, cv2, numpy as np
from jamo import h2j, j2hcj
from PIL import Image, ImageFont, ImageDraw

DEFAULT_IMAGE_DIR = '../../images/target'
DEFAULT_OUTPUT_DIR = '../../images/target-split-1type'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def separate_1type(img_dir, output_dir):
    
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

        vowel_1 = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ']
        #vowel_2 = ['ㅗ','ㅛ','ㅜ','ㅠ','ㅡ']
        #vowel_3 = ['ㅘ','ㅙ','ㅚ','ㅝ','ㅞ','ㅟ','ㅢ']
        
        if jamo_dict[j][1] in vowel_1 and len(jamo_dict[j]) == 2:
            # 초성: ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅍ ㅎ
            # 중성: ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅣ
            
            # image 불러와서 findContours
            image = cv2.imread(images_list[j])
            image_copy = image.copy()
            img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            
            center_points = {}
            center_points_dict = []
            sorted_center_points = []
            
            # len(contours) 갯수만큼 반복
            for i in range(1, len(contours)):

                # contours[i]를 감싼 bbox 생성
                x,y,w,h = cv2.boundingRect(contours[i])
        
                # contours[i] 의 중앙값 계산
                # dictionary 형태로 저장 > key는 i, value는 contours의 중앙값
                center_x = x + w // 2
                center_y = y + h // 2
                center_points[i] = (center_x, center_y)
                center_points_dict.append((i, center_points[i]))
                
            # x값 기준으로 오름차순 정렬
            sorted_center_points = sorted(center_points_dict, key=lambda x: x[1][0])
        
            # x값 기준으로 오름차순 정렬한 인덱스 가져오기
            sorted_contours_indices = [index for index, _ in sorted_center_points]
                
            # 게,테,폐
            if jamo_dict[j][1] == 'ㅔ' or jamo_dict[j][1] == 'ㅖ':
                
                # 게,AC8C
                if jamo_dict[j][0] == 'ㄱ':
                        
                    # 초성과 중성이 분리되어있고 ㅔ의 컨투어가 두 개인 경우
                    # 5,10,16,19,27,31,33,37,50,53,57,77,79,83,84
                    if fontname == '5' or fontname == '10' or fontname == '16' \
                    or fontname == '19' or fontname == '27' or fontname == '31' \
                    or fontname == '33' or fontname == '37' or fontname == '50' \
                    or fontname == '53' or fontname == '57' or fontname == '77' \
                    or fontname == '79' or fontname == '83' or fontname == '84':
                    
                        initial_component = contours[sorted_contours_indices[0]]
                        middle_component_1 = contours[sorted_contours_indices[1]]
                        middle_component_2 = contours[sorted_contours_indices[2]]
                            
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                            
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                            
                    # 초성과 중성이 붙어있는 경우
                    else:
                        pass
                            
                # 테, D14C
                elif jamo_dict[j][0] == 'ㅌ':
                        
                    # 초성의 컨투어가 2개인 경우 
                    # 5,7,18,22,24,28,33,36,37,42,43,44,49,55,56,59,65,76,84,85,91,
                    if fontname == '5' or fontname == '7' or fontname == '18' \
                    or fontname == '22' or fontname == '24' or fontname == '28' \
                    or fontname == '33' or fontname == '36' or fontname == '37' \
                    or fontname == '42' or fontname == '43' or fontname == '44' \
                    or fontname == '49' or fontname == '55' or fontname == '56' \
                    or fontname == '59' or fontname == '65' or fontname == '76' \
                    or fontname == '84' or fontname == '85' or fontname == '91':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        middle_component_1 = contours[sorted_contours_indices[2]]
                        middle_component_2 = contours[sorted_contours_indices[3]]
                            
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                            
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                    # 초성과 중성이 붙어있는 경우 
                    # 2,16,67,70,90,
                    elif fontname == '2' or fontname == '16' or fontname == '67' \
                    or fontname == '70' or fontname == '90' or fontname == '91':
                        pass
                        
                # 폐,D3D0
                elif jamo_dict[j][0] == 'ㅍ':
                    
                    # 초성 컨투어가 한 개인 경우 
                    # 23,30,31,32,38,39,40,45,47,48,53,65,69,70,72,74,80,85,87,88,97
                    if fontname == '23' or fontname == '30' or fontname == '31' \
                    or fontname == '32' or fontname == '38' or fontname == '39' \
                    or fontname == '40' or fontname == '45' or fontname == '47' \
                    or fontname == '48' or fontname == '53' or fontname == '65' \
                    or fontname == '69' or fontname == '70' or fontname == '72' \
                    or fontname == '74' or fontname == '80' or fontname == '85' or fontname == '87' \
                    or fontname == '88' or fontname == '97':
                        initial_component = contours[sorted_contours_indices[0]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)

                        # 중성 컨투어가 세 개인 경우 
                        # 72
                        if fontname == '72':
                            middle_component_1 = contours[sorted_contours_indices[1]]
                            middle_component_2 = contours[sorted_contours_indices[2]]
                            middle_component_3 = contours[sorted_contours_indices[3]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_3)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                        # 중성 컨투어가 두 개인 경우 
                        # 그 외
                        else:
                            middle_component_1 = contours[sorted_contours_indices[1]]
                            middle_component_2 = contours[sorted_contours_indices[2]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                    # 초성 컨투어가 두 개인 경우 
                    # 3,4,5,6,10,13,15,19,21,22,24,27,36,37,49,56,57,63,66,75,81,
                    elif fontname == '3' or fontname == '4' or fontname == '5' \
                    or fontname == '6' or fontname == '10' or fontname == '13' \
                    or fontname == '15' or fontname == '19' or fontname == '21' \
                    or fontname == '22' or fontname == '24' or fontname == '27' \
                    or fontname == '36' or fontname == '37' or fontname == '49' \
                    or fontname == '56' or fontname == '57' or fontname == '63' \
                    or fontname == '66' or fontname == '75' or fontname == '81':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        # 중성 컨투어가 세 개인 경우
                        # 22,56,63,
                        if fontname == '22' or fontname == '56' or fontname == '63':
                            middle_component_1 = contours[sorted_contours_indices[2]]
                            middle_component_2 = contours[sorted_contours_indices[3]]
                            middle_component_3 = contours[sorted_contours_indices[4]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_3)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                        # 중성 컨투어가 4개인 경우
                        # 81
                        elif fontname == '81':
                            middle_component_1 = contours[sorted_contours_indices[2]]
                            middle_component_2 = contours[sorted_contours_indices[3]]
                            middle_component_3 = contours[sorted_contours_indices[4]]
                            middle_component_4 = contours[sorted_contours_indices[5]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_3)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_4)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                        
                        # 중성 컨투어가 두 개인 경우 
                        # 그 외
                        else:
                            middle_component_1 = contours[sorted_contours_indices[2]]
                            middle_component_2 = contours[sorted_contours_indices[3]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                    # 초성 컨투어가 세 개인 경우 
                    # 7,11,16,28,33,42,46,54,58,68,
                    elif fontname == '7' or fontname == '11' or fontname == '16' \
                    or fontname == '28' or fontname == '33' or fontname == '42' \
                    or fontname == '46' or fontname == '54' or fontname == '58' \
                    or fontname == '68':
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
                        
                        # 중성 컨투어가 세 개인 경우 
                        # 7,11,33,42,46
                        if fontname == '7' or fontname == '11' or fontname == '33' \
                        or fontname == '42' or fontname == '46':
                            middle_component_1 = contours[sorted_contours_indices[3]]
                            middle_component_2 = contours[sorted_contours_indices[4]]
                            middle_component_3 = contours[sorted_contours_indices[5]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_3)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                        else:
                            middle_component_1 = contours[sorted_contours_indices[3]]
                            middle_component_2 = contours[sorted_contours_indices[4]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)

                    # 초성과 중성이 붙어있는 경우 
                    # 55,59,62,67,90,94,
                    elif fontname == '55' or fontname == '59' or fontname == '62' \
                    or fontname == '67' or fontname == '90' or fontname == '94':
                        pass
                    
                    # 초성 컨투어가 안에 한 개, 밖에 한 개인 경우
                    # 그 외
                    else:
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1,initial_component_2])
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                    
                        middle_component_1 = contours[sorted_contours_indices[2]]
                        middle_component_2 = contours[sorted_contours_indices[3]]
                            
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)

            else:
                # 랴,져,커
                
                # 랴,B7B4
                if jamo_dict[j][0] == 'ㄹ':
                    
                    # 초성과 중성이 붙어있는 경우 
                    # 7,10,13,28,
                    if fontname == '7' or fontname == '10' or fontname == '13' \
                    or fontname == '28':
                        pass
                    
                    # 초성 컨투어가 두 개인 경우 
                    # 49,50,88,90
                    elif fontname == '49' or fontname == '50' or fontname == '88' \
                    or fontname == '90':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        middle_component = contours[sorted_contours_indices[2]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)

                    # 초성 컨투어가 한 개인 경우 
                    # 그 외
                    else:
                        initial_component = contours[sorted_contours_indices[0]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        # 중성 컨투어가 두 개인 경우 
                        # 11,16,
                        if fontname == '11' or fontname == '16':
                            
                            middle_component_1 = contours[sorted_contours_indices[1]]
                            middle_component_2 = contours[sorted_contours_indices[2]]
                        
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)

                        # 중성 컨투어가 한 개인 경우 
                        # 그외 
                        else:
                            middle_component = contours[sorted_contours_indices[1]]
                        
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                # 져,C838
                elif jamo_dict[j][0] == 'ㅈ':
                    
                    # 초성 컨투어가 두 개인 경우
                    # 7,11,32,33,37,
                    if fontname == '7' or fontname == '11' \
                    or fontname == '33' \
                    or fontname == '37':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        # 중성 컨투어가 두 개인 경우
                        # 7,11,32,33,
                        if fontname == '7' or fontname == '11' \
                        or fontname == '33':
                            middle_component_1 = contours[sorted_contours_indices[2]]
                            middle_component_2 = contours[sorted_contours_indices[3]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)

                        # 중성 컨투어가 한 개인 경우 
                        # 그 외
                        else:
                            middle_component = contours[sorted_contours_indices[2]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                    # 초성과 중성이 붙어있는 경우 
                    # 55,62,75,90
                    elif fontname == '55' or fontname == '62' or fontname == '75' \
                    or fontname == '90':
                        pass
                    
                    # 초성 컨투어가 한 개인 경우
                    # 그 외
                    else:
                        initial_component = contours[sorted_contours_indices[0]]
     
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        # 중성 컨투어가 두 개인 경우
                        if fontname == '25' or fontname == '27' or fontname == '29' \
                        or fontname == '30' or fontname == '32' or fontname == '39' \
                        or fontname == '42' or fontname == '46' or fontname == '52' \
                        or fontname == '54' \
                        or fontname == '57' or fontname == '58' or fontname == '61' \
                        or fontname == '63' or fontname == '67' or fontname == '68' \
                        or fontname == '74' or fontname == '80' or fontname == '94':
                            middle_component_1 = contours[sorted_contours_indices[1]]
                            middle_component_2 = contours[sorted_contours_indices[2]]
                                
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                        # 중성 컨투어가 세 개인 경우 
                        # 81
                        elif fontname == '81':
                            middle_component_1 = contours[sorted_contours_indices[1]]
                            middle_component_2 = contours[sorted_contours_indices[2]]
                            middle_component_3 = contours[sorted_contours_indices[3]]
                                
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(middle_component_3)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                        
                        else:
                            middle_component = contours[sorted_contours_indices[1]]
                                
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                    
                    
                # 커,CEE4
                elif jamo_dict[j][0] == 'ㅋ':
                    
                    # 초성과 중성이 붙어있는 경우 
                    # 7,11,14,17,20,23,24,25,26,28,30,34,35,38,44,47,48,
                    # 49,52,54,55,56,58,59,60,61,64,66,67,69,70,71,73,74,75,76,78,
                    # 80,82,86,87,89,90,91,96,97
                    if fontname == '7' or fontname == '11' or fontname == '14' \
                    or fontname == '17' or fontname == '20' or fontname == '23' \
                    or fontname == '24' or fontname == '25' or fontname == '26' \
                    or fontname == '28' or fontname == '30' or fontname == '34' \
                    or fontname == '35' or fontname == '38' or fontname == '44' \
                    or fontname == '47' or fontname == '48' or fontname == '49' \
                    or fontname == '52' or fontname == '54' or fontname == '55' \
                    or fontname == '56' or fontname == '58' or fontname == '59' \
                    or fontname == '60' or fontname == '61' or fontname == '64' \
                    or fontname == '66' or fontname == '67' or fontname == '69' \
                    or fontname == '70' or fontname == '71' or fontname == '73' \
                    or fontname == '74' or fontname == '75' or fontname == '76' \
                    or fontname == '78' or fontname == '80' or fontname == '82' \
                    or fontname == '86' or fontname == '87' or fontname == '89' \
                    or fontname == '90' or fontname == '91' or fontname == '96' \
                    or fontname == '97':
                        pass
                    
                    # 초성 컨투어가 두 개인 경우 
                    # 5,
                    elif fontname == '5':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        middle_component = contours[sorted_contours_indices[2]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                            
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                    # 초성 컨투어 한 개, 중성 컨투어 한 개인 경우
                    else:
                        initial_component = contours[sorted_contours_indices[0]]
                        middle_component = contours[sorted_contours_indices[1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                            
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)

                
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-dir', type=str, dest='img_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR)
   
    args = parser.parse_args()

    separate_1type(args.img_dir, args.output_dir)