import argparse, glob, io, os, cv2, numpy as np
from jamo import h2j, j2hcj
from PIL import Image, ImageFont, ImageDraw

DEFAULT_IMAGE_DIR = '../../images/target'
DEFAULT_OUTPUT_DIR = '../../images/target-split-6type'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def separate_6type(img_dir, output_dir):
    
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
    
        if jamo_dict[j][1] in vowel_3 and len(jamo_dict[j]) == 3:
            
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
                    
            # 관, 꿠, 봤, 퓥, 횤
            
            # 관, AD00
            if jamo_dict[j][0]=='ㄱ':
                
                # 초성, 중성이 붙어있는 경우 
                # 11,20,24,26,27,35,36,48,56,66,68,69,72,75,78,89,94,97
                if fontname == '11' or fontname == '20' or fontname == '24' or fontname == '26' \
                or fontname == '27' or fontname == '35' or fontname == '36' or fontname == '48' \
                or fontname == '56' or fontname == '66' or fontname == '68' or fontname == '69' \
                or fontname == '72' or fontname == '75' or fontname == '78' or fontname == '89' \
                or fontname == '94' or fontname == '97' or fontname == '98':
                    pass     
                
                # 초성,중성,종성이 잘 분리된 경우 (그 외)
                else:
                    initial_component = contours[sorted_contours_indices[0]]
                    final_component = contours[sorted_contours_indices[-1]]
                            
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
            
            
            # 꿠, AFE0
            elif jamo_dict[j][0] == 'ㄲ':
                
                # 초성 컨투어 두 개, 종성 컨투어 두 개인 경우 
                # 3,4,14,31,33,34,42,50,51,57,73,83,84,92,95
                if fontname == '3' or fontname == '4' or fontname == '14' or fontname == '31' \
                or fontname == '33' or fontname == '34' or fontname == '42' or fontname == '50' \
                or fontname == '51' or fontname == '57' or fontname == '73' or fontname == '83' \
                or fontname == '84' or fontname == '92' or fontname == '95':
                    initial_component_1 = contours[sorted_contours_indices[0]]
                    initial_component_2 = contours[sorted_contours_indices[1]]
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)

                # 초성 컨투어 한 개, 종성 컨투어 두 개인 경우 
                # 7,17,21,27,29,39,41,45,46,52,62,63,65,66,69,74,76,77,85,86,88,
                elif fontname == '7' or fontname == '17' or fontname == '21' \
                or fontname == '27' or fontname == '29' or fontname == '39' or fontname == '41' \
                or fontname == '45' or fontname == '46' or fontname == '52' or fontname == '62' \
                or fontname == '63' or fontname == '65' or fontname == '66' or fontname == '69' \
                or fontname == '74' or fontname == '76' or fontname == '77' or fontname == '85' \
                or fontname == '86' or fontname == '88':
                    initial_component = contours[sorted_contours_indices[0]]
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                # 초성 컨투어 한 개, 종성 컨투어 한 개인 경우
                # 22,36,61,82,
                elif fontname == '22' or fontname == '36' or fontname == '61' or fontname == '82':
                    initial_component = contours[sorted_contours_indices[0]]
                    final_component = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                
                # 초성 컨투어 두 개, 종성 컨투어 한 개인 경우
                # 32,
                elif fontname == '32':
                    initial_component_1 = contours[sorted_contours_indices[0]]
                    initial_component_2 = contours[sorted_contours_indices[1]]
                    final_component = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                # 초성 컨투어 한 개, 종성 컨투어 세 개인 경우 
                # 11
                elif fontname == '11':
                    initial_component = contours[sorted_contours_indices[0]]
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component_3 = contours[sorted_contours_indices[-3]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_3)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    
                
                # 초성 컨투어 두 개, 종성 컨투어 세 개인 경우
                # 37,
                elif fontname == '37':
                    initial_component_1 = contours[sorted_contours_indices[0]]
                    initial_component_2 = contours[sorted_contours_indices[1]]
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component_3 = contours[sorted_contours_indices[-3]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_1.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                        
                    cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                    cv2.fillPoly(image_copy,[np.array(final_component_3)],(255,255,255))
                    file_string = f'{fontname}_{char_unicode}_2.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,image_copy)
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
            
            # 봤, BD24
            elif jamo_dict[j][0] == 'ㅂ':
                
                # 초성,중성,종성이 잘 분리된 경우 
                # 15,17,31,46,57,73,83,86,91
                if fontname == '15' or fontname == '17' or fontname == '29' \
                or fontname == '31' or fontname == '34' or fontname == '46' or fontname == '50' \
                or fontname == '57' or fontname == '60' or fontname == '64' or fontname == '73' \
                or fontname == '83' or fontname == '84' \
                or fontname == '86' \
                or fontname == '91':
                    
                    # 초성 ㅂ 컨투어가 안에 한 개 밖에 한 개이고 종성 컨투어가 한 개인 경우 
                    # 15,17,18,31,34,60,64,73,83,86,91,
                    if fontname == '15' or fontname == '17' or fontname == '18' \
                    or fontname == '31' or fontname == '34' or fontname == '60' \
                    or fontname == '64' or fontname == '73' or fontname == '83' \
                    or fontname == '86' or fontname == '91':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        final_component = contours[sorted_contours_indices[-1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                            
                        cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                        cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,image_copy)
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)

                    # 초성 ㅂ 컨투어가 안에 한 개 밖에 한 개이고 종성 컨투어가 두 개인 경우 
                    # 29,46,50,57,84,
                    elif fontname == '29' or fontname == '46' or fontname == '50' \
                    or fontname == '57' or fontname == '84':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        final_component_1 = contours[sorted_contours_indices[-1]]
                        final_component_2 = contours[sorted_contours_indices[-2]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                            
                        cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                        cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                        cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,image_copy)
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
            
            # 퓥, D4E5
            elif jamo_dict[j][0] == 'ㅍ':
                
                # 중성과 종성이 붙어있는 경우 
                # 12,14,19,20,23,26,28,35,36,40,41,48,49,53,54,55,56,61,63,64,67,68,69
                # 70,71,75,77,78,80,81,89,90,94,96,97,98
                if fontname == '12' or fontname == '14' or fontname == '19' or fontname == '20' \
                or fontname == '23' or fontname == '26' or fontname == '28' or fontname == '35' \
                or fontname == '36' or fontname == '40' or fontname == '41' or fontname == '48' \
                or fontname == '49' or fontname == '53' or fontname == '54' or fontname == '55' \
                or fontname == '56' or fontname == '61' or fontname == '63' or fontname == '64' \
                or fontname == '67' or fontname == '68' or fontname == '69' or fontname == '70' \
                or fontname == '71' or fontname == '75' or fontname == '77' or fontname == '78' \
                or fontname == '80' or fontname == '81' or fontname == '89' or fontname == '90' \
                or fontname == '94' or fontname == '96' or fontname == '97' or fontname == '98':
                    pass
                
                
                # 중성과 종성이 잘 분리된 경우(그 외)
                else:
                    
                    # 초성 ㅍ 컨투어가 한 개인 경우 
                    # 24,33,34,38,44,59,74,85,88,91,
                    if fontname == '24' or fontname == '33' or fontname == '34' \
                    or fontname == '38' or fontname == '44' or fontname == '59' \
                    or fontname == '74' or fontname == '85' or fontname == '88' \
                    or fontname == '91':
                        initial_component = contours[sorted_contours_indices[0]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        # 종성 ㅌ 컨투어가 한 개인 경우 
                        # 33,34,38,59,74,88,91
                        if fontname == '33' or fontname == '34' or fontname == '38' \
                        or fontname == '59' or fontname == '74' or fontname == '88' \
                        or fontname == '91':
                            final_component = contours[sorted_contours_indices[-1]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)

                        # 종성 ㅌ 컨투어가 두 개인 경우
                        # 24,44,85,
                        else:
                            final_component_1 = contours[sorted_contours_indices[-1]]
                            final_component_2 = contours[sorted_contours_indices[-2]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)
                            
                    
                    # 초성 ㅍ 컨투어가 두 개인 경우 
                    # 3,4,5,6,7,10,11,13,15,27,31,32,58,84
                    elif fontname == '3' or fontname == '4' or fontname == '5' or fontname == '6' \
                    or fontname == '7' or fontname == '10' or fontname == '11' or fontname == '13' \
                    or fontname == '15' or fontname == '27' or fontname == '31' or fontname == '32' \
                    or fontname == '58' or fontname == '84' or fontname == '89':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        # 종성 컨투어가 두 개인 경우
                        # 5,11,58,84,
                        if fontname == '5' or fontname == '11' or fontname == '58' or fontname == '84':
                            final_component_1 = contours[sorted_contours_indices[-1]]
                            final_component_2 = contours[sorted_contours_indices[-2]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)
                        
                        
                        # 종성 컨투어가 한 개인 경우 (그 외)
                        else:
                            final_component = contours[sorted_contours_indices[-1]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)
                            
                        
                                            
                    # 초성 ㅍ 컨투어가 세 개인 경우 
                    # 42,46,57,
                    elif fontname == '42' or fontname == '46' or fontname == '57':
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
                        
                        # 종성 컨투어가 세 개인 경우 
                        # 42 
                        if fontname == '42':
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
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(initial_component_3)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_3)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)
                        
                        # 종성 컨투어가 두 개인 경우
                        # 46,57
                        else:
                            final_component_1 = contours[sorted_contours_indices[-1]]
                            final_component_2 = contours[sorted_contours_indices[-2]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(initial_component_3)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)

                    # 초성 ㅍ 컨투어가 네 개인 경우 
                    # 16
                    elif fontname == '16':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component_3 = contours[sorted_contours_indices[2]]
                        initial_component_4 = contours[sorted_contours_indices[3]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_3)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_4)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        final_component = contours[sorted_contours_indices[-1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(255,255,255))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(255,255,255))
                        cv2.fillPoly(mask,[np.array(initial_component_3)],(255,255,255))
                        cv2.fillPoly(mask,[np.array(initial_component_4)],(255,255,255))
                        cv2.fillPoly(mask,[np.array(final_component)],(255,255,255))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                    
                    # 초성 ㅍ 컨투어가 안에 한 개, 밖에 한 개인 경우 (그 외)
                    else:
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        # 종성 컨투어가 세 개인 경우 
                        # 37, 
                        if fontname == '37':
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
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_3)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)
                        
                        # 종성 컨투어가 두 개인 경우
                        # 18,39,66,87
                        elif fontname == '18' or fontname == '39' or fontname == '66' or fontname == '87':
                            final_component_1 = contours[sorted_contours_indices[-1]]
                            final_component_2 = contours[sorted_contours_indices[-2]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)
                        
                        # 종성 컨투어가 한 개인 경우(그 외)
                        else:
                            final_component = contours[sorted_contours_indices[-1]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)

            # 횤, D6A4
            elif jamo_dict[j][0] == 'ㅎ':
                
                #  초성,중성,종성이 잘 분리된 경우 
                # 15,22,27,66,74,76,84,
                if fontname == '15' or fontname == '22' or fontname == '27' or fontname == '66' \
                or fontname == '74' or fontname == '76' or fontname == '84':
                    
                    # 초성 ㅎ 컨투어가 삐침 한개, ㅇ 안에 한 개, 밖에 한 개인 경우 
                    # 15,22,27,66,84
                    if fontname == '15' or fontname == '22' or fontname == '27' \
                    or fontname == '66' or fontname == '84':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_3 = contours[sorted_contours_indices[1]]
                        initial_component_4 = contours[sorted_contours_indices[2]]
                        initial_component_2 = np.vstack([initial_component_3,initial_component_4])
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        final_component = contours[sorted_contours_indices[-1]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                        cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                        cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,image_copy)

                    # 초성 ㅎ 컨투어가 안에 한 개, 밖에 한 개인 경우 
                    # 74,76
                    elif fontname == '74' or fontname == '76':
                        initial_component_3 = contours[sorted_contours_indices[0]]
                        initial_component_4 = contours[sorted_contours_indices[1]]
                        initial_component_2 = np.vstack([initial_component_3,initial_component_4])
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(initial_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        if fontname == '76':
                            
                            final_component = contours[sorted_contours_indices[-1]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)
                            
                        else:
                            final_component_1 = contours[sorted_contours_indices[-1]]
                            final_component_2 = contours[sorted_contours_indices[-2]]
                            
                            mask = np.zeros((256,256,3), np.uint8) + 255
                            cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                            cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                            file_string = f'{fontname}_{char_unicode}_3.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,mask)
                            
                            cv2.fillPoly(image_copy,[np.array(initial_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(initial_component_2)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_1)],(255,255,255))
                            cv2.fillPoly(image_copy,[np.array(final_component_2)],(255,255,255))
                            file_string = f'{fontname}_{char_unicode}_2.png'
                            file_path = os.path.join(output_dir, file_string)
                            cv2.imwrite(file_path,image_copy)
            
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-dir', type=str, dest='img_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR)
   
    args = parser.parse_args()

    separate_6type(args.img_dir, args.output_dir)