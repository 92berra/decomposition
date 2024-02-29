import argparse, glob, io, os, cv2, numpy as np
from jamo import h2j, j2hcj
from PIL import Image, ImageFont, ImageDraw

DEFAULT_IMAGE_DIR = '../../images/target'
DEFAULT_OUTPUT_DIR = '../../images/target-split-4type'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def separate_4type(img_dir, output_dir):
    
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
    
        if jamo_dict[j][1] in vowel_1 and len(jamo_dict[j]) == 3:
            
            # image 불러와서 findContours
            image = cv2.imread(images_list[j])
            image_copy = image.copy()
            img_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                        
            center_points_dict = []
            center_points = {}
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
                        
            # y좌표 기준으로 오름차순 정렬
            sorted_center_points = sorted(center_points_dict, key=lambda x: x[1][1])
                        
            # y좌표 기준으로 오름차순 정렬한 인덱스 가져오기
            sorted_contours_indices = [index for index, _ in sorted_center_points]
                    
            # 4유형의 초성: ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅍ ㅎ
            # 4유형의 중성: ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅣ
            # 4유형의 종성: ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅄ ㅅ ㅆ ㅇ ㅈ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ
            
            # 긴, AE34
            if jamo_dict[j][0]=='ㄱ':
                
                final_component = contours[sorted_contours_indices[2]]
                            
                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_3.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)
                
                # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                        
                # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                
                initial_component = contours[sorted_contours_indices_2[0]]
                middle_component = contours[sorted_contours_indices_2[1]]
                            
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
                
            # 깎, AE4E
            elif jamo_dict[j][0]=='ㄲ':
                
                # 종성의 ㄲ 붙어있는 경우 
                # 7,11,17,18,21,22,23,24,26,29,30,32,36,38,40,41
                # 47,48,50,52,55,56,58,59,60,61,62,65,66,68,69,72
                # 73,74,78,79,80,82,85,86,87,93,94,96,97
                if fontname == '7' or fontname == '11' or fontname == '17' \
                or fontname =='18' or fontname == '21' or fontname == '22' \
                or fontname =='23' or fontname == '24' or fontname == '26' \
                or fontname =='29' or fontname == '30' or fontname == '32' \
                or fontname =='36' or fontname == '38' or fontname == '40' \
                or fontname =='41' or fontname == '47' or fontname == '48' \
                or fontname =='50' or fontname == '52' or fontname == '55' \
                or fontname =='56' or fontname == '58' or fontname == '59' \
                or fontname =='60' or fontname == '61' or fontname == '62' \
                or fontname =='65' or fontname == '66' or fontname == '68' \
                or fontname =='69' or fontname == '72' or fontname == '73' \
                or fontname =='74' or fontname == '78' or fontname == '79' \
                or fontname =='80' or fontname == '82' or fontname == '85' \
                or fontname =='86' or fontname == '87' or fontname == '93' \
                or fontname =='94' or fontname == '96' or fontname == '97':
                
                    final_component = contours[sorted_contours_indices[-1]]
                                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                            
                    # 초성의 ㄲ이 떨어져있는 경우
                    # 29,36,38,40,52,94
                    if fontname == '29' or fontname == '36' or fontname == '38' \
                    or fontname == '40' or fontname == '52' or fontname == '72' \
                    or fontname == '94':
                             
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[2]]
                                        
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
                        
                    # 초성의 ㄲ이 붙어있는 경우
                    else:
                        
                        # 중성이 2개의 컨투어인 경우
                        # 74
                        if fontname == '74':
                            initial_component = contours[sorted_contours_indices_2[0]]
                            middle_component_1 = contours[sorted_contours_indices_2[1]]
                            middle_component_2 = contours[sorted_contours_indices_2[2]]
                                            
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
                            
                        else:
                            initial_component = contours[sorted_contours_indices_2[0]]
                            middle_component = contours[sorted_contours_indices_2[1]]
                                            
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
                    
                    
                # 종성의 ㄲ 이 떨어져있는 경우
                # 1,2,3,4,5,6,8,9,10,12,13,14,15,16,19,25,27,28,31,33
                # 34,35,37,42,45,46,49,51,54,57,63,70,75,76,77,83,84,
                # 88,89,91,92,95,98
                if fontname == '1' or fontname == '2' or fontname == '3' \
                or fontname == '4' or fontname == '5' or fontname == '6' \
                or fontname == '8' or fontname == '9' or fontname == '10' \
                or fontname == '12' or fontname == '13' or fontname == '14' \
                or fontname == '15' or fontname == '16' or fontname == '19' \
                or fontname == '25' or fontname == '27' or fontname == '28' \
                or fontname == '31' or fontname == '33' or fontname == '34' \
                or fontname == '35' or fontname == '37' or fontname == '42' \
                or fontname == '45' or fontname == '46' or fontname == '49' \
                or fontname == '51' or fontname == '54' or fontname == '57' \
                or fontname == '63' or fontname == '70' or fontname == '75' \
                or fontname == '76' or fontname == '77' or fontname == '83' \
                or fontname == '84' or fontname == '88' or fontname == '89' \
                or fontname == '91' or fontname == '92' or fontname == '95' \
                or fontname == '98':
                    
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # 초성의 ㄲ이 붙어있는 경우
                    # 3,10,13,14,15,25,28,35,45,46,49,54,63,
                    # 70,76,77,89,91,92
                    if fontname == '3' or fontname == '10' or fontname == '13' \
                    or fontname == '14' or fontname == '15' or fontname == '25' \
                    or fontname == '28' or fontname == '35' or fontname == '45' \
                    or fontname == '46' or fontname == '49' or fontname == '54' \
                    or fontname == '63' or fontname == '70' or fontname == '76' \
                    or fontname == '77' or fontname == '89' or fontname == '91' \
                    or fontname == '92':

                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                                        
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
                        
                    
                    # 초성의 ㄲ이 떨어져있는 경우
                    else: 
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[2]]
                                        
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
                    
                # 종성의 ㄲ 이 세 개의 컨투어인 경우 
                # 39
                if fontname == '39':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component_3 = contours[sorted_contours_indices[-3]]
                                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                            
                    initial_component = contours[sorted_contours_indices_2[0]]
                    middle_component = contours[sorted_contours_indices_2[1]]
                                        
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
                
                # 중성과 종성이 붙어있는 경우
                # 20,53,64,67,71,81,90
                # 무시
                
            # 넋, B10B
            elif jamo_dict[j][0]=='ㄴ':
                
                # 종성의 ㄱ과 ㅅ이 붙어있는 경우
                # 10,59,60,67,68,70,94
                if fontname == '10' or fontname == '59' or fontname == '60' \
                or fontname == '67' or fontname == '68' or fontname == '70' \
                or fontname == '94':
                    final_component = contours[sorted_contours_indices[-1]]
                                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # 중성이 2개의 컨투어인 경우
                    # 68,70
                    if fontname == '68' or fontname == '70':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component_1 = contours[sorted_contours_indices_2[1]]
                        middle_component_2 = contours[sorted_contours_indices_2[2]]
                                            
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
                    
                    # 중성이 1개의 컨투어인 경우
                    else:        
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                                            
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
                
                # 종성의 획이 세 개인 경우
                # 7,11,37
                elif fontname == '7' or fontname == '11' or fontname == '37':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component_3 = contours[sorted_contours_indices[-3]]
                                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                            
                    # 중성이 2개의 컨투어인 경우
                    # 7
                    if fontname == '7':
                        
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component_1 = contours[sorted_contours_indices_2[1]]
                        middle_component_2 = contours[sorted_contours_indices_2[2]]
                                            
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
                        
                    else:
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                                            
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
                        
                
                # 종성과 중성이 붙은 경우 
                # 54,81,90
                elif fontname == '54' or fontname == '81' or fontname == '90':
                    pass
                
                # 종성의 ㄱ과 ㅅ이 떨어져있는 경우
                else:
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                        
                    # 중성이 2개의 컨투어인 경우
                    # 56,75
                    if fontname == '56' or fontname == '75':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component_1 = contours[sorted_contours_indices_2[1]]
                        middle_component_2 = contours[sorted_contours_indices_2[2]]
                                  
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
                        
                    else:
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                                            
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
                    
            # 닭, B2ED
            elif jamo_dict[j][0]=='ㄷ':
                
                # 종성의 ㄹ과 ㄱ이 붙어있는 경우 
                # 59,69,80,97
                if fontname == '59' or fontname == '69' or fontname == '80' \
                or fontname == '97':
                    final_component = contours[sorted_contours_indices[-1]]
                                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                        
                    initial_component = contours[sorted_contours_indices_2[0]]
                    middle_component = contours[sorted_contours_indices_2[1]]
                                            
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
                
                # 종성과 중성이 붙어있는 경우
                # 78,81
                elif fontname == '78' or fontname == '81':
                    pass
                
                # 초성과 중성이 붙어있는 경우
                # 7,10,13,20,22,26,28,35,48,49,55,56,89
                elif fontname == '7' or fontname == '10' or fontname == '13' \
                or fontname == '20' or fontname == '22' or fontname == '26' \
                or fontname == '28' or fontname == '35' or fontname == '48' \
                or fontname == '49' or fontname == '55' or fontname == '56' \
                or fontname == '89':
                    pass
                
                # 종성의 ㄹ과 ㄱ이 분리된 경우
                else:
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                        
                    # 초성이 2개의 컨투어인 경우
                    # 16,33,42,85,91
                    if fontname == '16' or fontname == '33' or fontname == '42' \
                    or fontname == '85' or fontname == '91':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[2]]
                                                
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
                        
                    # 초성이 1개의 컨투어, 중성이 2개의 컨투어인 경우 
                    elif fontname == '37':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component_1 = contours[sorted_contours_indices_2[1]]
                        middle_component_2 = contours[sorted_contours_indices_2[2]]
                                                
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
                        
                    else:
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                                                
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

            # 떻,B5BB
            elif jamo_dict[j][0]=='ㄸ':
                
                # 종성 ㅎ의 삐침과 ㅇ이 붙어서 2개의 컨투어인 경우 
                # 1,11,16,19,48,68,74,93,98
                if fontname == '1' or fontname == '11' or fontname == '16' \
                or fontname == '19' or fontname == '48' or fontname == '68' \
                or fontname == '74' or fontname == '93' or fontname == '98':
                    
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component = np.vstack([final_component_1, final_component_2])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    middle_component = contours[sorted_contours_indices_2[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                
                # 종성 ㅎ의 삐침이 붙어있고 ㅇ과 분리된 경우 
                # 2,5,7,12,14,15,18,21,22,23,24,25,29,32,35,38
                # 40,44,45,49,52,53,57,59,65,76,78,79,80,83,88,
                # 95,
                elif fontname == '2' or fontname == '5' or fontname == '7' \
                or fontname == '12' or fontname == '14' or fontname == '15' \
                or fontname == '18' or fontname == '21' or fontname == '22' \
                or fontname == '23' or fontname == '24' or fontname == '25' \
                or fontname == '29' or fontname == '32' or fontname == '35' \
                or fontname == '38' or fontname == '40' or fontname == '44' \
                or fontname == '45' or fontname == '49' or fontname == '52' \
                or fontname == '53' or fontname == '57' or fontname == '59' \
                or fontname == '65' or fontname == '76' or fontname == '78' \
                or fontname == '79' or fontname == '80' or fontname == '83' \
                or fontname == '88' or fontname == '95':
                    
                    final_component_1 = contours[sorted_contours_indices[-3]]
                    final_component_3 = contours[sorted_contours_indices[-2]]
                    final_component_4 = contours[sorted_contours_indices[-1]]
                    final_component_2 = np.vstack([final_component_3, final_component_4])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    middle_component = contours[sorted_contours_indices_2[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                
                # 종성 ㅎ의 삐침이 분리되어있고 삐침과 ㅇ이 분리되어있는 경우
                # 3,4,8,9,10,13,17,30,31,33,34,36,37,39,42,46,47,50,51,
                # 55,60,61,62,63,66,73,77,81,82,86,87,89,90,91,94,97,
                elif fontname == '3' or fontname == '4' or fontname == '8' \
                or fontname == '9' or fontname == '10' or fontname == '13' \
                or fontname == '17' or fontname == '30' or fontname == '31' \
                or fontname == '33' or fontname == '34' or fontname == '36' \
                or fontname == '37' or fontname == '39' or fontname == '42' \
                or fontname == '46' or fontname == '47' or fontname == '50' \
                or fontname == '51' or fontname == '55' or fontname == '60' \
                or fontname == '61' or fontname == '62' or fontname == '63' \
                or fontname == '66' or fontname == '73' or fontname == '77' \
                or fontname == '81' or fontname == '82' or fontname == '86' \
                or fontname == '87' or fontname == '89' or fontname == '90' \
                or fontname == '91' or fontname == '94' or fontname == '97':
                    final_component_1 = contours[sorted_contours_indices[-4]]
                    final_component_2 = contours[sorted_contours_indices[-3]]
                    final_component_4 = contours[sorted_contours_indices[-2]]
                    final_component_5 = contours[sorted_contours_indices[-1]]
                    final_component_3 = np.vstack([final_component_4,final_component_5])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-4], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    middle_component = contours[sorted_contours_indices_2[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_3)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                
                # 종성 ㅎ의 삐침이 하나만 분리되어있고 나머지 삐침은 ㅇ과 붙어있는 경우
                # 6,20,41,43,54,56,69,70,75,84,96
                elif fontname == '6' or fontname == '20' or fontname == '41' \
                or fontname == '43' or fontname == '54' or fontname == '56' \
                or fontname == '69' or fontname == '70' or fontname == '75' \
                or fontname == '84' or fontname == '96':
                    final_component_1 = contours[sorted_contours_indices[-3]]
                    final_component_3 = contours[sorted_contours_indices[-2]]
                    final_component_4 = contours[sorted_contours_indices[-1]]
                    final_component_2 = np.vstack([final_component_3,final_component_4])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    middle_component = contours[sorted_contours_indices_2[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                
                # 종성 ㅎ의 삐침이 분리되어있고 나머지 삐침과 ㅇ이 하나의 컨투어인 경우
                # 27
                elif fontname == '27':
                    final_component_1 = contours[sorted_contours_indices[-2]]
                    final_component_2 = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    middle_component = contours[sorted_contours_indices_2[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                
                # 종성 ㅎ의 삐침이 붙어있고 ㅇ이 하나의 컨투어인 경우 
                # 85,92
                elif fontname == '85' or fontname == '92':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # x좌표 기준으로 오름차순 정렬 (마지막 요소 제외하고)
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                                    
                    # x좌표 기준으로 오름차순 정렬한 인덱스 가져오기
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    middle_component = contours[sorted_contours_indices_2[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                
                # 중성과 종성이 붙어있는 경우 
                # 26,28,58,64,67,71,72,
                elif fontname == '26' or fontname == '28' or fontname == '58' \
                or fontname == '64' or fontname == '67' or fontname == '71' \
                or fontname == '72':
                    pass
                
            # 많,B9CE
            elif jamo_dict[j][0]=='ㅁ':
                
                # 종성의 ㄴ과 ㅎ이 떨어져있고 ㅎ의 컨투어가 2개인 경우
                # 1,5,11,25,48,68,74,93,
                if fontname == '1' or fontname == '5' or fontname == '11' \
                or fontname == '25' or fontname == '48' or fontname == '68' \
                or fontname == '74' or  fontname == '93':

                    # len(contours) == 3
                    # y값 기준으로 오름차순 정렬 한 결과에서 마지막 3개만 가져와서 x기준 오름차순 정렬
                    sorted_center_points_final_component = sorted(sorted_center_points[-3:], key=lambda x: x[1][0])
                    sorted_contours_indices_final_component = [index for index, _ in sorted_center_points_final_component]
                    
                    final_component_1 = contours[sorted_contours_indices_final_component[0]]
                    final_component_3 = contours[sorted_contours_indices_final_component[1]]
                    final_component_4 = contours[sorted_contours_indices_final_component[2]]
                    final_component_2 = np.vstack([final_component_3, final_component_4])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # ㅁ이 하나의 컨투어인 경우
                    # 11,68
                    # len(contours) == 2
                    if fontname == '11' or fontname == '68':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                        
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
                    
                    else:
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                        
                # 종성의 ㄴ과 ㅎ이 떨어져있고 ㅎ의 삐침획이 하나의 컨투어이고 삐침과 ㅇ이 분리된 경우
                # 2,12,14,15,18,21,22,23,29,38,40,45,51,52,65,73,76,78,79,82,83,87,88,92,94,95,
                elif fontname == '2' or fontname == '12' or fontname == '14' \
                or fontname == '15' or fontname == '18' or fontname == '21' or fontname == '22' \
                or fontname == '23' or fontname == '29' or fontname == '38' or fontname == '40' \
                or fontname == '45' or fontname == '51' or fontname == '52' or fontname == '65' \
                or fontname == '73' or fontname == '76' or fontname == '78' or fontname == '79' \
                or fontname == '82' or fontname == '83' or fontname == '87' or fontname == '88' \
                or fontname == '92' or fontname == '94' or fontname == '95':
                    
                    # len(contours) == 5
                    # y값 기준으로 오름차순 정렬 한 결과에서 마지막 네 개만 x기준 오름차순 정렬
                    sorted_center_points_final_component = sorted(sorted_center_points[-4:], key=lambda x: x[1][0])
                    sorted_contours_indices_final_component = [index for index, _ in sorted_center_points_final_component]
                    
                    final_component_1 = contours[sorted_contours_indices_final_component[-4]]
                    final_component_2 = contours[sorted_contours_indices_final_component[-3]]
                    final_component_4 = contours[sorted_contours_indices_final_component[-2]]
                    final_component_5 = contours[sorted_contours_indices_final_component[-1]]
                    final_component_3 = np.vstack([final_component_4, final_component_5])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-4], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                        
                    initial_component_1 = contours[sorted_contours_indices_2[0]]
                    initial_component_2 = contours[sorted_contours_indices_2[1]]
                    initial_component = np.vstack([initial_component_1, initial_component_2])
                    middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                    

                # 종성의 ㄴ과 ㅎ이 떨어져있고 ㅎ의 삐침획이 두 개의 컨투어이고 ㅇ도 떨어져있는 경우 
                # 3,4,8,9,17,26,27,28,30,31,33,34,36,37,39,42,44,46,47,50,53,57,60,61,62,63,77,80,86,
                elif fontname == '3' or fontname == '4' or fontname == '8' \
                or fontname == '9' or fontname == '17' \
                or fontname == '30' or fontname == '31' or fontname == '33' \
                or fontname == '34' or fontname == '36' or fontname == '37' or fontname == '39' \
                or fontname == '42' or fontname == '44' or fontname == '46' or fontname == '47' \
                or fontname == '50' or fontname == '53' or fontname == '57' or fontname == '60' \
                or fontname == '61' or fontname == '62' or fontname == '63' or fontname == '77' \
                or fontname == '80' or fontname == '86':
                    # len(contours)==5
                    sorted_center_points_final_component = sorted(sorted_center_points[-5:], key=lambda x: x[1][0])
                    sorted_contours_indices_final_component = [index for index, _ in sorted_center_points_final_component]
                    
                    final_component_1 = contours[sorted_contours_indices_final_component[0]]
                    final_component_2 = contours[sorted_contours_indices_final_component[1]]
                    final_component_3 = contours[sorted_contours_indices_final_component[2]]
                    final_component_5 = contours[sorted_contours_indices_final_component[3]]
                    final_component_6 = contours[sorted_contours_indices_final_component[4]]
                    final_component_4 = np.vstack([final_component_5, final_component_6])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_4)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-5], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # ㅁ이 하나의 컨투어인 경우
                    # 33,
                    if fontname == '33':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                        
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
                    
                    # ㅁ이 두 개의 컨투어인 경우
                    # 42
                    elif fontname == '42':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                        
                    # ㅁ의 안의 컨투어와 밖의 컨투어가 있는 경우
                    else:
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        middle_component = contours[sorted_contours_indices_2[-1]]
                        
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
                    
                # 종성의 ㄴ과 ㅎ이 떨어져있고 ㅎ의 삐침획이 떨어져있고 나머지 획과 ㅇ이 붙어있는 경우 
                # 6,16,26,27,41,43,54,56,66,75,84,89,91,96,
                # len(contours)==4
                if fontname == '6' or fontname == '16' or fontname == '26' or fontname == '27' \
                or fontname == '41' or fontname == '43' or fontname == '54' or fontname == '56' \
                or fontname == '66' or fontname == '75' or fontname == '84' or fontname == '89' \
                or fontname == '91' or fontname == '96':
                    sorted_center_points_final_component = sorted(sorted_center_points[-4:], key=lambda x: x[1][0])
                    sorted_contours_indices_final_component = [index for index, _ in sorted_center_points_final_component]
                    
                    final_component_1 = contours[sorted_contours_indices_final_component[0]]
                    final_component_2 = contours[sorted_contours_indices_final_component[1]]
                    final_component_4 = contours[sorted_contours_indices_final_component[2]]
                    final_component_5 = contours[sorted_contours_indices_final_component[3]]
                    final_component_3 = np.vstack([final_component_4, final_component_5])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-4], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                
                    # ㅁ이 하나의 컨투어인 경우
                    # 16,28,75,89
                    if fontname == '16' or fontname == '28' or fontname == '75' or fontname == '89':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                        
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
                        
                    else:
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component = np.vstack([initial_component_1,initial_component_2])
                        middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                        
                
                # 종성의 ㄴ과 ㅎ이 떨어져있고 ㅎ의 ㅇ이 하나의 컨투어인 경우
                # 85,
                if fontname == '85':
                    sorted_center_points_final_component = sorted(sorted_center_points[-3:], key=lambda x: x[1][0])
                    sorted_contours_indices_final_component = [index for index, _ in sorted_center_points_final_component]
                    
                    final_component_1 = contours[sorted_contours_indices_final_component[0]]
                    final_component_2 = contours[sorted_contours_indices_final_component[1]]
                    final_component_3 = contours[sorted_contours_indices_final_component[2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    initial_component_1 = contours[sorted_contours_indices_2[0]]
                    initial_component_2 = contours[sorted_contours_indices_2[1]]
                    initial_component = np.vstack([initial_component_1, initial_component_2])
                    middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                
                # 종성의 ㄴ과 ㅎ의 ㅇ이 붙어있고 ㅎ의 삐침이 두 개의 컨투어인 경우
                # 10,13,97
                if fontname == '10' or fontname == '13':
                    sorted_center_points_final_component = sorted(sorted_center_points[-3:], key=lambda x: x[1][0])
                    sorted_contours_indices_final_component = [index for index, _ in sorted_center_points_final_component]
                    
                    final_component_1 = contours[sorted_contours_indices_final_component[0]]
                    final_component_2 = contours[sorted_contours_indices_final_component[1]]
                    final_component_3 = contours[sorted_contours_indices_final_component[2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    initial_component_1 = contours[sorted_contours_indices_2[0]]
                    initial_component_2 = contours[sorted_contours_indices_2[1]]
                    initial_component = np.vstack([initial_component_1, initial_component_2])
                    middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                    

                # 종성의 ㄴ과 ㅎ의 ㅇ이 붙어있고 ㅎ의 삐침이 하나의 컨투어인 경우 
                # 49,67,
                # elif fontname == '49' or fontname == '67':
                
                # 종성의 ㄴ과 ㅎ의 ㅇ이 붙어있고 ㅎ의 삐침과 ㅇ이 붙어있고 ㅇ의 컨투어가 두 개인 경우
                # 19,59
                # elif fontname == '19' or fontname == '59':
                    
                    # ㅁ이 두 개의 컨투어인 경우
                    # 19
                    # if fontname == '19':
                    
                    # else:
                
                # 종성의 ㄴ과 ㅎ의 ㅇ이 붙어있고 ㅎ의 삐침이 맨 위만 분리되어있고 나머지 삐침은 ㅇ과 붙어있는 경우
                # 69,90,
                # elif fontname == '69' or fontname == '90':
                
                # 종성의 ㄴ과 ㅎ의 두 번째 삐침이 붙어있고 첫번째 삐침은 분리되어있고 ㅇ의 컨투어가 두 개인 경우 
                # 24
                # elif fontname == '24':
                
                # 중성과 종성이 붙어있는 경우
                # 7,20,32,35,55,58,64,70,71,72,81,98
                elif fontname == '7' or fontname == '20' or fontname == '32' \
                or fontname == '35' or fontname == '55' or fontname == '58' \
                or fontname == '64' or fontname == '70' or fontname == '71' \
                or fontname == '72' or fontname == '81' or fontname == '97' or fontname == '98':
                    pass
                
            # 뱀, BC40
            elif jamo_dict[j][0]=='ㅂ':
            
                # 종성의 컨투어가 한 개인 경우
                # 7,11,56,63,66,68,70,74,89,
                if fontname == '7' or fontname == '11' or fontname == '56' \
                or fontname == '63' or fontname == '66' or fontname == '68' \
                or fontname == '70' or fontname == '74' or fontname == '89':
                    final_component = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # 초성의 컨투어가 한 개인 경우
                    # 89,
                    if fontname == '89':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                        
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
                     
                    # 초성의 컨투어가 따로 한 개, 나머지 두 개가 안에 한 개 밖에 한개인 경우
                    # 7,11,
                    elif fontname == '7' or fontname == '11':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_3 = contours[sorted_contours_indices_2[1]]
                        initial_component_4 = contours[sorted_contours_indices_2[2]]
                        initial_component_2 = np.vstack([initial_component_3,initial_component_4])
                        middle_component = contours[sorted_contours_indices_2[3]]
                        
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

                    # 초성의 컨투어가 안에 한 개, 밖에 한 개인 경우
                    else:
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component = np.vstack([initial_component_1,initial_component_2])
                        middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                    
                # 종성의 컨투어가 두 개인 경우
                # 19,28,42,
                elif fontname == '19' or fontname == '28' or fontname == '42':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]

                    # 초성의 컨투어가 두 개인 경우 
                    # 19
                    if fontname == '19':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[-1]]
                        
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
                        
                    else:
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component = np.vstack([initial_component_1,initial_component_2])
                        middle_component = contours[sorted_contours_indices_2[-1]]
                        
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
                    
                # 중성과 종성이 붙어있는 경우
                # 40,53,60,64,67,78,81,90,94,
                elif fontname == '40' or fontname == '53' or fontname == '60' \
                or fontname == '64' or fontname == '67' or fontname == '78' \
                or fontname == '81' or fontname == '90' or fontname == '94':
                    pass
                
                # 초성과 중성이 붙어있는 경우 
                # 97
                elif fontname == '97':
                    pass
                
                # 종성의 컨투어가 안에 한 개, 밖에 한 개인 경우
                else:
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component = np.vstack([final_component_1,final_component_2])
                        
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]

                    # 초성의 컨투어가 한 개인 경우 
                    # 5,27,
                    if fontname == '5' or fontname == '27':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                            
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
                        
                    else:
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component = np.vstack([initial_component_1,initial_component_2])
                        middle_component = contours[sorted_contours_indices_2[-1]]
                            
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
                    
            # 집, C9D1
            elif jamo_dict[j][0]=='ㅈ':
                
                # 종성 ㅂ이 한 개의 컨투어인 경우 
                # 5,23,66,70,86,
                if fontname == '5' or fontname == '23' or fontname == '37' or fontname == '66' \
                or fontname == '70' or fontname == '86':
                    final_component = contours[sorted_contours_indices[-1]]
                        
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # 초성이 두 개의 컨투어인 경우
                    # 37
                    if fontname == '37':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[-1]]
                        
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
                       
                    # 그 외의 경우 
                    else:
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[-1]]
                        
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
                        
                # 종성 ㅂ이 한 개의 컨투어가 분리되고 안의 컨투어, 밖의 컨투어인 경우 
                # 7,11,42,48,
                elif fontname == '7' or fontname == '11' or fontname == '42' or fontname == '48':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_3 = contours[sorted_contours_indices[-2]]
                    final_component_4 = contours[sorted_contours_indices[-3]]
                    final_component_2 = np.vstack([final_component_3, final_component_4])
                        
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]

                    # 초성이 두 개의 컨투어인 경우 
                    # 11
                    if fontname == '11':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[-1]]
                        
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
                        
                    else:
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[-1]]
                        
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
                
                # 종성 ㅂ이 두 개의 컨투어인 경우 
                # 41,85,
                elif fontname == '41' or fontname == '85':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    initial_component = contours[sorted_contours_indices_2[0]]
                    middle_component = contours[sorted_contours_indices_2[-1]]
                        
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

                # 중성과 종성이 붙어있는 경우 
                # 18,19,20,25,26,28,40,44,53,54,55,56,58,59,64,67,68,71,72,78,80,81,89,93,94,96,
                elif fontname == '18' or fontname == '19' or fontname == '20' or fontname == '25' \
                or fontname == '26' or fontname == '28' or fontname == '40' or fontname == '44' \
                or fontname == '53' or fontname == '54' or fontname == '55' or fontname == '56' \
                or fontname == '58' or fontname == '59' or fontname == '64' or fontname == '67' \
                or fontname == '68' or fontname == '71' or fontname == '72' or fontname == '78' \
                or fontname == '80' or fontname == '81' or fontname == '89' or fontname == '93' \
                or fontname == '94' or fontname == '96':
                    pass
                
                # 초성 ㅈ과 종성 ㅂ이 붙어있는 경우 
                # 49,
                elif fontname == '49':
                    pass
                
                # 종성의 ㅂ이 안의 컨투어, 밖의 컨투어인 경우
                else:
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component = np.vstack([final_component_1,final_component_2])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]

                    # 초성의 컨투어가 두 개인 경우 
                    # 16,
                    if fontname == '16':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[-1]]
                            
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
                        
                    else:
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[-1]]
                            
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
                
            # 쨟, CA1F
            elif jamo_dict[j][0]=='ㅉ':
                
                # 종성의 ㄹ 컨투어가 한 개, ㅂ 컨투어가 한 개인 경우 
                # 5,19,27,35,84,
                if fontname == '5' or fontname == '19' or fontname == '27' \
                or fontname == '35' or fontname == '84':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # 초성 컨투어가 하나인 경우
                    # 5,19,
                    if fontname == '5' or fontname == '19':
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                        
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

                    # 초성 컨투어가 두 개인 경우
                    # 27,84
                    elif fontname == '27' or fontname == '84':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                        
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

                    # 초성 컨투어가 안에 한 개, 밖에 한 개인 경우 
                    # 35,
                    elif fontname == '35':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component = np.vstack([initial_component_1,initial_component_2])
                        middle_component = contours[sorted_contours_indices_2[1]]
                        
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
                
                # 종성의 ㄹ 컨투어와 ㅂ의 획 한개가 붙고 ㅂ 나머지 컨투어가 안과 밖으로 된 경우
                # 7,
                
                # 종성의 ㄹ 컨투어가 한 개, ㅂ 컨투어가 두 개인 경우
                # 11,
                
                # 종성의 ㄹ과 ㅂ이 붙어있고 ㅂ 컨투어가 안과 밖인 경우
                # 24,25,26,32,38,43,44,48,55,60,71,
                
                # 중성과 종성이 붙어있는 경우
                # 28,59,61,67,75,78,
                
                # 종성의 ㄹ과 ㅂ이 붙어있어 ㄹ이 안과 밖, ㅂ이 안과 밖의 컨투어인 경우 
                # 54,69,
        
                
            # 핥, D565
            elif jamo_dict[j][0]=='ㅎ':
                
                # 종성의 ㄹ 컨투어가 한 개, ㅌ 컨투어가 두 개인 경우 
                # 5,16,18,36,37,65,72,73,76,84,90,
                if fontname == '5' or fontname == '16' or fontname == '18' \
                or fontname == '36' or fontname == '37' or fontname == '65' \
                or fontname == '72' or fontname == '73' or fontname == '76' \
                or fontname == '84' or fontname == '90':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component_3 = contours[sorted_contours_indices[-3]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # 초성의 ㅎ에서 삐침 컨투어가 한 개, ㅇ 컨투어가 안에 한 개, 밖에 한 개인 경우
                    # 5,16,18,37,84,
                    if fontname == '5' or fontname == '16' or fontname == '18' \
                    or fontname == '37' or fontname == '84':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_3 = contours[sorted_contours_indices_2[1]]
                        initial_component_4 = contours[sorted_contours_indices_2[2]]
                        initial_component_2 = np.vstack([initial_component_3,initial_component_4])
                        middle_component = contours[sorted_contours_indices_2[3]]
                        
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
                    
                    # 초성의 ㅎ에서 삐침 컨투어가 두 개, ㅇ 컨투어가 안에 한 개, 밖에 한 개인 경우 
                    # 36,65,72,73,90
                    elif fontname == '36' or fontname == '65' or fontname == '72' \
                    or fontname == '73' or fontname == '90':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component_4 = contours[sorted_contours_indices_2[2]]
                        initial_component_5 = contours[sorted_contours_indices_2[3]]
                        initial_component_3 = np.vstack([initial_component_4,initial_component_5])
                        middle_component = contours[sorted_contours_indices_2[4]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_3)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                    # 초성의 ㅎ에서 삐침과 ㅇ이 붙어있고 ㅇ 컨투어가 안에 한 개, 밖에 한 개인 경우
                    # 76,
                    elif fontname == '76':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        initial_component = np.vstack([initial_component_1,initial_component_2])
                        middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                        
                                 
                # 종성과 중성이 붙어있는 경우 
                # 20,67,81,
                elif fontname == '20' or fontname == '67' or fontname == '81':
                    pass
                
                # 종성의 ㄹ과 ㅌ이 붙어있는 경우 
                # 24,25,28,32,35,40,43,48,68,69,70,78,96,
                elif fontname == '24' or fontname == '25' or fontname == '28' \
                or fontname == '32' or fontname == '35' or fontname == '40' \
                or fontname == '43' or fontname == '48' or fontname == '68' \
                or fontname == '69' or fontname == '70' or fontname == '78' \
                or fontname == '96':
                    pass
                
                # 종성의 ㄹ과 ㅌ이 붙어있고 ㅌ의 컨투어가 두 개인 경우
                # 26,42,55,
                elif fontname == '26' or fontname == '42' or fontname == '55':
                    pass
                
                # 종성의 ㄹ과 ㅌ의 컨투어가 안에 한 개, 밖에 한 개인 경우 
                # 38, 54,60,71,80,82,97,
                elif fontname == '38' or fontname == '54' or fontname == '60' \
                or fontname == '71' or fontname == '80' or fontname == '82' \
                or fontname == '97':
                    pass
                
                # 종성의 ㄹ 컨투어가 한 개, ㅌ 컨투어가 안에 한 개, 밖에 한 개인 경우 
                # 45,49,79,
                elif fontname == '45' or fontname == '49' or fontname == '79':
                    pass
                
                # 초성과 중성이 붙어있는 경우 
                # 75,
                elif fontname == '75':
                    pass
                
                # 종성의 ㄹ 컨투어가 한 개, ㅌ 컨투어가 한 개인 경우 (그 외 모두 해당)
                # else:
                
            # 실, C2E4
            elif jamo_dict[j][0]=='ㅅ' and jamo_dict[j][2]=='ㄹ':
                final_component = contours[sorted_contours_indices[-1]]
                    
                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_3.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)
                        
                sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                
                # 초성 컨투어가 두 개인 경우
                # 7,33,37,81,
                if fontname == '7' or fontname == '33' or fontname == '37' or fontname == '81':
                    initial_component_1 = contours[sorted_contours_indices_2[0]]
                    initial_component_2 = contours[sorted_contours_indices_2[1]]
                    middle_component = contours[sorted_contours_indices_2[2]]
                    
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

                # 초성 컨투어가 한 개인 경우 (그 외 모두 해당)
                else:
                    initial_component = contours[sorted_contours_indices_2[0]]
                    middle_component = contours[sorted_contours_indices_2[-1]]
                    
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

            # 싫, C2EB
            #elif jamo_dict[j][0]=='ㅅ' and jamo_dict[j][2]=='ㅀ':
            elif char_unicode == 'C2EB':
                
                # 종성의 ㄹ 컨투어가 한 개, ㅎ 컨투어가 안에 한 개, 밖에 한 개인 경우
                # 1,5,7,11,
                if fontname == '1' or fontname == '5' or fontname == '7' or fontname == '11':
                    
                    # 종성의 ㄹ과 ㅎ을 x좌표 기준으로 정렬
                    sorted_center_points_3 = sorted(sorted_center_points[-3:], key=lambda x: x[1][0])
                    sorted_contours_indices_3 = [index for index, _ in sorted_center_points_3]
                    
                    final_component_1 = contours[sorted_contours_indices_3[0]] # ㄹ
                    final_component_3 = contours[sorted_contours_indices_3[1]]
                    final_component_4 = contours[sorted_contours_indices_3[2]]
                    final_component_2 = np.vstack([final_component_3,final_component_4])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                            
                    sorted_center_points_2 = sorted(sorted_center_points[:-3], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # 초성 컨투어가 두 개인 경우
                    # 7,11
                    if fontname == '7' or fontname == '11':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                        
                    
                    # 초성 컨투어가 한 개인 경우(그 외)
                    else:
                        initial_component = contours[sorted_contours_indices_2[0]]
                        middle_component = contours[sorted_contours_indices_2[1]]
                        
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
                        
                                    
                # 종성의 ㄹ 컨투어가 한 개, ㅎ 컨투어가 삐침 한 개, 나머지 삐침이 ㅇ과 붙어있고 ㅇ 컨투어는 안에 한 개, 밖에 한 개인 경우
                # 6,16,
                # elif fontname == '6' or fontname == '16':
                
                # 종성의 ㄹ 컨투어가 한 개, ㅎ 컨투어가 삐침 두 개, ㅇ 컨투어가 안에 한 개, 밖에 한개 인 경우 
                # 3,4,8,9,
                # elif fontname == '3' or fontname == '4' or fontname == '8' or fontname == '9':
                
                # 종성의 ㄹ 컨투어가 ㅎ의 ㅇ과 붙어있고 ㅎ 컨투어가 삐침 두 개, ㅇ 컨투어가 안에 한 개, 밖에 한 개인 경우
                # 10,
                # elif fontname == '10':
                
                # 종성의 ㄹ 컨투어가 한 개, ㅎ의 삐침 컨투어가 한 개, ㅇ 컨투어가 안에 한 개, 밖에 한 개인 경우
                # 2,12,13,14,15,
                # elif fontname == '2' or fontname == '12' or fontname == '13' or fontname == '14' or fontname == '15':
                
                # 종성의 ㄹ 컨투어가 ㅎ의 두번째 삐침과 붙어있고 ㅎ의 맨 위 삐침 컨투어 한 개, ㅇ의 컨투어가 안에 한 개, 밖에 한 개인 경우
                # 13, 
                # elif fontname == '13':
                
            # 쌕, C315
            elif jamo_dict[j][0]=='ㅆ' and jamo_dict[j][2]=='ㄱ':
                
                final_component = contours[sorted_contours_indices[-1]]
                    
                mask = np.zeros((256,256,3), np.uint8) + 255
                cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                file_string = f'{fontname}_{char_unicode}_3.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,mask)
                        
                sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                
                # 초성과 중성이 붙어있는 경우 
                # 28,94,97,
                if fontname == '28' or fontname == '94' or fontname == '97':
                    pass
                
                # 중성과 종성이 붙어있는 경우
                # 36,40,64,68,78,81,89,90,
                elif fontname == '36' or fontname == '40' or fontname == '64' \
                or fontname == '68' or fontname == '78' or fontname == '81' \
                or fontname == '89' or fontname == '90':
                    pass
                
                # 초성의 컨투어가 두 개인 경우 
                # 3,11,37,42,50,52,57,63,83,84,88,
                elif fontname == '3' or fontname == '11' or fontname == '37' \
                or fontname == '42' or fontname == '50' or fontname == '52' \
                or fontname == '57' or fontname == '63' or fontname == '83' \
                or fontname == '84' or fontname == '88':
                    
                    initial_component_1 = contours[sorted_contours_indices_2[0]]
                    initial_component_2 = contours[sorted_contours_indices_2[1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                
                    # 중성의 컨투어가 두 개인 경우
                    # 37
                    if fontname == '37':
                        middle_component_1 = contours[sorted_contours_indices_2[-2]]
                        middle_component_2 = contours[sorted_contours_indices_2[-1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                    else:
                        middle_component = contours[sorted_contours_indices_2[-1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)

                # 초성의 컨투어가 한 개인 경우 (그 외)
                else:
                    initial_component = contours[sorted_contours_indices_2[0]]
                    middle_component = contours[sorted_contours_indices_2[1]]
                    
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

            # 썢, C362
            elif jamo_dict[j][0]=='ㅆ' and jamo_dict[j][2]=='ㅈ':
                
                # 종성 컨투어가 두 개인 경우
                # 16,37,
                if fontname == '16' or fontname == '37':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    if fontname == '37':
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component = contours[sorted_contours_indices_2[2]]
                        
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
                        
                    else:
                        initial_component_1 = contours[sorted_contours_indices_2[0]]
                        initial_component_2 = contours[sorted_contours_indices_2[1]]
                        middle_component_1 = contours[sorted_contours_indices_2[2]]
                        middle_component_2 = contours[sorted_contours_indices_2[3]]
                        middle_component = np.vstack([middle_component_1, middle_component_2])
                        
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

                # 종성과 중성이 붙어있는 경우
                # 5,56,64,
                elif fontname == '5' or fontname == '56' or fontname == '64':
                    pass
                
                # 초성과 중성이 붙어있는 경우 
                # 26,33,54,55,67,70,75,76,78,87,94,95,97,
                elif fontname == '26' or fontname == '33' or fontname == '54' \
                or fontname == '55' or fontname == '67' or fontname == '70' \
                or fontname == '75' or fontname == '76' or fontname == '78' \
                or fontname == '87' or fontname == '94' or fontname == '95' \
                or fontname == '97':
                    pass
                
                # 초성과 중성과 종성이 붙어있는 경우
                # 28,68,
                elif fontname == '28' or fontname == '68':
                    pass
                
                # 종성 컨투어가 한 개인 경우 (그 외)
                else:
                    pass
                    # final_component = contours[sorted_contours_indices[-1]]
                    
                    # mask = np.zeros((256,256,3), np.uint8) + 255
                    # cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    # file_string = f'{fontname}_{char_unicode}_3.png'
                    # file_path = os.path.join(output_dir, file_string)
                    # cv2.imwrite(file_path,mask)
                    
                    # sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                    # sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    # # 초성 컨투어가 두 개인 경우
                    # # 3,4,42,50,52,57,63,81,83,84,88,92,
                    # if fontname == '3' or fontname == '4' or fontname == '7' or fontname == '21' \
                    # or fontname == '42' or fontname == '50' or fontname == '52' \
                    # or fontname == '57' or fontname == '63' or fontname == '81' \
                    # or fontname == '83' or fontname == '84' or fontname == '88' \
                    # or fontname == '92':
                        
                    #     initial_component_1 = contours[sorted_contours_indices_2[0]]
                    #     initial_component_2 = contours[sorted_contours_indices_2[1]]
                        
                    #     mask = np.zeros((256,256,3), np.uint8) + 255
                    #     cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    #     cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    #     file_string = f'{fontname}_{char_unicode}_1.png'
                    #     file_path = os.path.join(output_dir, file_string)
                    #     cv2.imwrite(file_path,mask)

                    #     # 중성 컨투어가 두 개인 경우 
                    #     # 42
                    #     if fontname == '42':
                    #         middle_component_1 = contours[sorted_contours_indices_2[2]]
                    #         middle_component_2 = contours[sorted_contours_indices_2[3]]
                            
                    #         mask = np.zeros((256,256,3), np.uint8) + 255
                    #         cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                    #         cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                    #         file_string = f'{fontname}_{char_unicode}_2.png'
                    #         file_path = os.path.join(output_dir, file_string)
                    #         cv2.imwrite(file_path,mask)
                            
                    #     # 중성 컨투어가 한 개인 경우
                    #     elif fontname == '11' or fontname == '81':
                    #         middle_component = contours[sorted_contours_indices_2[2]]
                            
                    #         mask = np.zeros((256,256,3), np.uint8) + 255
                    #         cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    #         file_string = f'{fontname}_{char_unicode}_2.png'
                    #         file_path = os.path.join(output_dir, file_string)
                    #         cv2.imwrite(file_path,mask)
                        
                        
                    #     # 중성 컨투어가 안에 한 개, 밖에 한 개인 경우 (그 외)
                    #     else:
                    #         middle_component_1 = contours[sorted_contours_indices_2[2]]
                    #         middle_component_2 = contours[sorted_contours_indices_2[3]]
                    #         middle_component = np.vstack([middle_component_1, middle_component_2])
                            
                    #         mask = np.zeros((256,256,3), np.uint8) + 255
                    #         cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    #         file_string = f'{fontname}_{char_unicode}_2.png'
                    #         file_path = os.path.join(output_dir, file_string)
                    #         cv2.imwrite(file_path,mask)
                        
                    # # 초성 컨투어가 한 개인 경우(그 외)
                    # else:
                    #     initial_component = contours[sorted_contours_indices_2[0]]
                        
                    #     mask = np.zeros((256,256,3), np.uint8) + 255
                    #     cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                    #     file_string = f'{fontname}_{char_unicode}_1.png'
                    #     file_path = os.path.join(output_dir, file_string)
                    #     cv2.imwrite(file_path,mask)

                    #     # 중성 컨투어가 한 개인 경우
                    #     # 93
                    #     if fontname == '93':
                    #         middle_component = contours[sorted_contours_indices_2[1]]
                            
                    #         mask = np.zeros((256,256,3), np.uint8) + 255
                    #         cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    #         file_string = f'{fontname}_{char_unicode}_2.png'
                    #         file_path = os.path.join(output_dir, file_string)
                    #         cv2.imwrite(file_path,mask)
                            
                    #     else:
                    #         middle_component_1 = contours[sorted_contours_indices_2[1]]
                    #         middle_component_2 = contours[sorted_contours_indices_2[2]]
                    #         middle_component = np.vstack([middle_component_1, middle_component_2])
                            
                    #         mask = np.zeros((256,256,3), np.uint8) + 255
                    #         cv2.fillPoly(mask,[np.array(middle_component)],(0,0,0))
                    #         file_string = f'{fontname}_{char_unicode}_2.png'
                    #         file_path = os.path.join(output_dir, file_string)
                    #         cv2.imwrite(file_path,mask)

            # 앉, C549
            elif jamo_dict[j][0]=='ㅇ' and jamo_dict[j][2]=='ㄵ':
            #elif char_unicode == 'C549':
                
                # 종성의 ㄴ과 ㅈ이 붙어있는 경우 
                # 10,13,49,68,75,97,
                if fontname == '10' or fontname == '13' or fontname == '49' or fontname == '68' \
                or fontname == '75' or fontname == '97':
                    
                    final_component = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-1], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    initial_component_1 = contours[sorted_contours_indices_2[0]]
                    initial_component_2 = contours[sorted_contours_indices_2[1]]
                    initial_component = np.vstack([initial_component_1, initial_component_2])
                    middle_component = contours[sorted_contours_indices_2[2]]
                    
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

                # 종성의 ㄴ 컨투어가 한 개, ㅈ 컨투어가 두 개인 경우
                # 1,6,10,16,22,37,
                
                    # 초성 컨투어가 한 개인 경우
                    # 
                
                # 종성의 ㄴ 컨투어가 한 개, ㅈ컨투어가 한 개인 경우 (그 외)
                    # 초성의 ㅇ 컨투어가 한 개인 경우 
                    # 7
                    
                # 중성과 종성이 붙어있는 경우 
                # 19,20,35,48,53,55,58,59,64,67,71,78,81,90,

            
            # 엾, C5FE
            elif jamo_dict[j][0]=='ㅇ' and jamo_dict[j][2]=='ㅄ':
                
                # 종성의 ㅂ 컨투어가 한 개, ㅅ 컨투어가 한 개인 경우 
                # 5,27
                if fontname == '5' or fontname == '27':
                    
                    # 종성의 ㅂ과 ㅅ을 x좌표 기준으로 정렬
                    sorted_center_points_3 = sorted(sorted_center_points[-3:], key=lambda x: x[1][0])
                    sorted_contours_indices_3 = [index for index, _ in sorted_center_points_3]
                    
                    final_component_1 = contours[sorted_contours_indices_3[-1]]
                    final_component_2 = contours[sorted_contours_indices_3[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    sorted_center_points_2 = sorted(sorted_center_points[:-2], key=lambda x: x[1][0])
                    sorted_contours_indices_2 = [index for index, _ in sorted_center_points_2]
                    
                    initial_component_1 = contours[sorted_contours_indices_2[0]]
                    initial_component_2 = contours[sorted_contours_indices_2[1]]
                    initial_component = np.vstack([initial_component_1, initial_component_2])
                    middle_component = contours[sorted_contours_indices_2[2]]
                    
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
                    
                    
                # 종성의 ㅂ 컨투어가 두 개, ㅅ 컨투어가 두 개인 경우 
                # 7,
                
                

                            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-dir', type=str, dest='img_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR)
   
    args = parser.parse_args()

    separate_4type(args.img_dir, args.output_dir)