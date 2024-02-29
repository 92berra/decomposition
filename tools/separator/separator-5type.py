import argparse, glob, io, os, cv2, numpy as np
from jamo import h2j, j2hcj
from PIL import Image, ImageFont, ImageDraw

DEFAULT_IMAGE_DIR = '../../images/target'
DEFAULT_OUTPUT_DIR = '../../images/target-split-5type'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def separate_5type(img_dir, output_dir):
    
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
    
        if jamo_dict[j][1] in vowel_2 and len(jamo_dict[j]) == 3:
            
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
                    
            # 굶, 뜻, 롶, 묻, 용, 읊, 줄, 쪽, 틈, 횿, 붓, 뽕
            
            # 굶, AD76
            if jamo_dict[j][0]=='ㄱ':
                
                # 종성과 중성이 붙어있는 경우
                # 2,7,
                if fontname == '2' or fontname == '7':
                    pass

                # 초성,중성,종성이 붙어있는 경우
                # 1,14,26,32,43,60,
                elif fontname == '1' or fontname == '14' or fontname == '26' or fontname == '32' \
                or fontname == '43' or fontname == '60':
                    pass
                
                # 중성과 초성이 붙어있는 경우
                # 8,9,10,
                elif fontname == '8' or fontname == '9' or fontname == '10':
                    pass
                
                # 초성과 중성과 종성이 붙어있는 경우
                # 6,11,12,13,20,35,44,54,61,80,87,90
                elif fontname == '6' or fontname == '11' or fontname == '12' or fontname == '13' \
                or fontname == '20' or fontname == '35' or fontname == '44' \
                or fontname == '54' or fontname == '61' or fontname == '80' \
                or fontname == '87' or fontname == '90':
                    pass
                
                # 초성과 중성이 붙어있는 경우 
                # 24,28,30,38,53,55,58,59,67,68,70,72,75,78,81,89,93,94,96,97,98
                elif fontname == '24' or fontname == '28' or fontname == '30' \
                or fontname == '38' or fontname == '53' or fontname == '55' \
                or fontname == '58' or fontname == '59' or fontname == '67' \
                or fontname == '68' or fontname == '70' or fontname == '72' \
                or fontname == '75' or fontname == '78' or fontname == '81' \
                or fontname == '89' or fontname == '93' or fontname == '94' \
                or fontname == '96' or fontname == '97' or fontname == '98':
                    pass
                
                # 종성의 ㄹ 컨투어가 한 개, ㅁ 컨투어가 한 개인 경우
                # 19,23,42,
                elif fontname == '19' or fontname == '23' or fontname == '42':
                    
                    # y값 기준으로 오름차순 정렬 한 결과에서 마지막 3개만 가져와서 x기준 오름차순 정렬
                    sorted_center_points_final_component = sorted(sorted_center_points[-2:], key=lambda x: x[1][0])
                    sorted_contours_indices_final_component = [index for index, _ in sorted_center_points_final_component]
                    
                    final_component_1 = contours[sorted_contours_indices_final_component[0]]
                    final_component_2 = contours[sorted_contours_indices_final_component[1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
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
                    
                
                # 종성의 ㄹ과 ㅁ이 붙어있고 ㄹ과 ㅁ 사이에 빈 컨투어가 있는 경우 
                # 69,
                elif fontname == '69':
                    final_component_1 = contours[sorted_contours_indices_final_component[-1]]
                    final_component_2 = contours[sorted_contours_indices_final_component[-2]]
                    final_component = np.vstack([final_component_1,final_component_2])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
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
                    
                
                # 종성의 ㄹ 컨투어가 한 개, 종성의 ㅁ 컨투어가 안에 한 개, 밖에 한 개인 경우 (그 외)
                else:
                    # len(contours) == 3
                    # y값 기준으로 오름차순 정렬 한 결과에서 마지막 3개만 가져와서 x기준 오름차순 정렬
                    sorted_center_points_final_component = sorted(sorted_center_points[-3:], key=lambda x: x[1][0])
                    sorted_contours_indices_final_component = [index for index, _ in sorted_center_points_final_component]
                    
                    final_component_1 = contours[sorted_contours_indices_final_component[0]]
                    final_component_3 = contours[sorted_contours_indices_final_component[1]]
                    final_component_4 = contours[sorted_contours_indices_final_component[2]]
                    final_component_2 = np.vstack([final_component_3,final_component_4])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
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
                    
            # 뜻, B73B
            elif jamo_dict[j][0]=='ㄸ':
                
                # 중성과 종성이 붙어있는 경우 
                # 7,19,24,26,37,96,97
                if fontname == '7' or fontname == '19' or fontname == '24' \
                or fontname == '26' or fontname == '37' or fontname == '96' \
                or fontname == '97':
                    pass
                
                # 종성의 ㅅ 컨투어가 두 개인 경우 
                # 16,49,81,
                elif fontname == '16' or fontname == '49' or fontname == '81':
                    
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)

                     # 초성의 ㄸ 컨투어가 두 개인 경우 
                     # 81
                    if fontname == '81':
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
                        
                    
                    # 초성의 ㄸ 컨투어가 세 개인 경우
                    # 16,49
                    elif fontname == '16' or fontname == '49':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component_3 = contours[sorted_contours_indices[2]]
                        middle_component = contours[sorted_contours_indices[3]]
                        
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
                        
                # 종성 컨투어가 한 개인 경우 (그 외)
                else:
                    final_component = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # 초성 ㄸ 컨투어가 안에 한 개, 밖에 한 개인 경우 
                    # 10,14,18,20,22,24,25,26,35,43,44,48,54,55,56,59,60,62,69,71,78,80,85,86,87,96,97,
                    if fontname == '10' or fontname == '14' or fontname == '18' or fontname == '20' \
                    or fontname == '22' or fontname == '24' or fontname == '25' \
                    or fontname == '26' or fontname == '35' or fontname == '43' \
                    or fontname == '44' or fontname == '48' or fontname == '54' \
                    or fontname == '55' or fontname == '56' or fontname == '59' \
                    or fontname == '60' or fontname == '62' or fontname == '69' or fontname == '71' \
                    or fontname == '78' or fontname == '80' or fontname == '85' \
                    or fontname == '86' or fontname == '87' or fontname == '96' \
                    or fontname == '97':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        middle_component = contours[sorted_contours_indices[2]]
                        
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

                    # 초성 ㄸ 컨투어가 한 개인 경우 
                    # 11,14,17,23,27,28,29,30,34,36,38,45,63,66,68,70,71,74,75,76,77,88,89,90,91,93,95,
                    elif fontname == '11' or fontname == '14' or fontname == '17' or fontname == '23'\
                    or fontname == '27' or fontname == '28' or fontname == '29' \
                    or fontname == '30' or fontname == '34' or fontname == '36' \
                    or fontname == '38' or fontname == '45' or fontname == '63' or fontname == '64' or fontname == '66' \
                    or fontname == '68' or fontname == '70' or fontname == '71' \
                    or fontname == '74' or fontname == '75' or fontname == '76' \
                    or fontname == '77' or fontname == '88' or fontname == '89' or fontname == '90' or fontname == '91' or fontname == '93' \
                    or fontname == '95':
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

                    # 초성 ㄸ 컨투어가 세 개인 경우 
                    # 16,49,58,90,
                    elif fontname == '16' or fontname == '33' or fontname == '49' or fontname == '58':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component_3 = contours[sorted_contours_indices[2]]
                        middle_component = contours[sorted_contours_indices[3]]
                        
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
                        
                    
                    # 초성 ㄸ 컨투어가 두 개인 경우(그 외)
                    else:
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        middle_component = contours[sorted_contours_indices[-2]]
                        
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
                        
            # 롶, B876
            elif jamo_dict[j][0]=='ㄹ':
                
                # 초성과 중성이 분리된 경우 
                # 15,22,34,42,46,57,60,81,83,84,91,
                if fontname == '15' or fontname == '22' or fontname == '34' \
                or fontname == '42' or fontname == '46' or fontname == '57' \
                or fontname == '60' or fontname == '81' or fontname == '83' \
                or fontname == '84' or fontname == '91':
                
                    # 종성의 컨투어가 한 개인 경우 
                    # 60,81,
                    if fontname == '60' or fontname == '81':
                        initial_component = contours[sorted_contours_indices[0]]
                        middle_component = contours[sorted_contours_indices[1]]
                        final_component = contours[sorted_contours_indices[2]]
                        
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
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)

                    # 종성의 컨투어가 두 개인 경우 
                    # 15,42,57,
                    elif fontname == '15' or fontname == '42' or fontname == '57':
                        initial_component = contours[sorted_contours_indices[0]]
                        middle_component = contours[sorted_contours_indices[1]]
                        final_component_1 = contours[sorted_contours_indices[2]]
                        final_component_2 = contours[sorted_contours_indices[3]]
                        
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
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                    
                    # 종성의 컨투어가 안에 한 개, 밖에 한 개인 경우 
                    # 22,34,46,83,84,91,
                    else:
                        initial_component = contours[sorted_contours_indices[0]]
                        middle_component = contours[sorted_contours_indices[1]]
                        final_component_1 = contours[sorted_contours_indices[2]]
                        final_component_2 = contours[sorted_contours_indices[3]]
                        final_component = np.vstack([final_component_1, final_component_2])
                        
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
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                else: 
                    pass
                
            # 묻, BB3B
            elif jamo_dict[j][0]=='ㅁ':
                
                # 중성과 종성이 분리된 경우 
                # 16,36,39,79,84,
                if fontname == '16' or fontname == '36' or fontname == '39' \
                or fontname == '79' or fontname == '84':
                    final_component = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # 초성의 ㅁ 컨투어가 한 개인 경우 
                    # 16
                    if fontname == '16':
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
                        
                    else:
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        middle_component = contours[sorted_contours_indices[2]]
                        
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
                        
            # 붓, BD93
            elif jamo_dict[j][0]=='ㅂ':
                
                # 종성과 중성이 붙어있는 경우 
                # 2,3,7,8,9,10,13,14,15,16,17,18,19,20,22,24,26,28,29,30,32,33,35,36,37,38,
                # 39,40,43,44,45,46,49,53,54,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70
                # 72,74,76,78,79,80,81,88,89,90,92,93,94,97
                if fontname == '2' or fontname == '3' or fontname == '7' or fontname == '8' \
                or fontname == '9' or fontname == '10' or fontname == '13' or fontname == '14' \
                or fontname == '15' or fontname == '16' or fontname == '17' or fontname == '18' \
                or fontname == '19' or fontname == '20' or fontname == '22' or fontname == '24' \
                or fontname == '26' or fontname == '28' or fontname == '29' or fontname == '30' \
                or fontname == '32' or fontname == '33' or fontname == '35' or fontname == '36' \
                or fontname == '37' or fontname == '38' or fontname == '39' or fontname == '40' \
                or fontname == '43' or fontname == '44' or fontname == '45' or fontname == '46' \
                or fontname == '49' or fontname == '53' or fontname == '54' or fontname == '56' \
                or fontname == '57' or fontname == '58' or fontname == '59' or fontname == '60' \
                or fontname == '61' or fontname == '62' or fontname == '63' or fontname == '64' \
                or fontname == '65' or fontname == '66' or fontname == '67' or fontname == '68' \
                or fontname == '69' or fontname == '70' or fontname == '72' or fontname == '74' \
                or fontname == '76' or fontname == '78' or fontname == '79' or fontname == '80' \
                or fontname == '81' or fontname == '88' or fontname == '89' or fontname == '90' \
                or fontname == '92' or fontname == '93' or fontname == '94' or fontname == '97':
                    pass
                
                # 종성 ㅅ 컨투어가 한 개인 경우(그 외)
                else:
                    final_component = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    # 초성 ㅂ 컨투어가 한 개인 경우
                    # 5,56
                    if fontname == '5' or fontname == '56':
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
                    
                    # 초성 ㅂ 컨투어가 두 개인 경우
                    # 86
                    elif fontname == '86':
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
                        
                    
                    # 초성 ㅂ 컨투어가 따로 한 개, 나머지 두 개가 안에 한개, 밖에 한 개인 경우 
                    # 11,42,
                    elif fontname == '11' or fontname == '42':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_3 = contours[sorted_contours_indices[1]]
                        initial_component_4 = contours[sorted_contours_indices[2]]
                        initial_component_2 = np.vstack([initial_component_3, initial_component_4])
                        middle_component = contours[sorted_contours_indices[3]]
                        
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
                        
                    
                    # 초성 ㅂ 컨투어가 안에 한 개, 밖에 한 개인 경우(그 외)
                    else:
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        middle_component = contours[sorted_contours_indices[2]]
                        
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
                        
            # 뽕, BF55
            elif jamo_dict[j][0]=='ㅃ':
                
                # 초성과 중성이 붙어있는 경우 
                # 1,5,6,7,10,12,13,14,20,23,25,26,30,35,36,
                # 38,39,40,43,44,48,53,54,55,58,64,69,71,73,
                # 75,77,82,85,88,97,98
                if fontname == '1' or fontname == '5' or fontname == '6' or fontname == '7' \
                or fontname == '10' or fontname == '12' or fontname == '13' or fontname == '14' \
                or fontname == '20' or fontname == '23' or fontname == '25' or fontname == '26' \
                or fontname == '30' or fontname == '35' or fontname == '36' or fontname == '38' \
                or fontname == '39' or fontname == '40' or fontname == '43' or fontname == '44' \
                or fontname == '48' or fontname == '53' or fontname == '54' or fontname == '55' \
                or fontname == '58' or fontname == '64' or fontname == '69' or fontname == '71' \
                or fontname == '73' or fontname == '75' or fontname == '77' or fontname == '82' \
                or fontname == '85' or fontname == '88' or fontname == '97' or fontname == '98':
                    pass
                
                # 초성, 중성, 종성이 분리된 경우 (그 외)
                else:
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component = np.vstack([final_component_1, final_component_2])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    middle_component = contours[sorted_contours_indices[-3]]
                    
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
                    
            # 용, C6A9
            elif jamo_dict[j][0]=='ㅇ' and jamo_dict[j][2]=='ㅇ':
                
                # 초성과 중성이 붙어있는 경우 
                # 1,6,7,8,9,10,11,12,13,14,20,23,24,28,29,35,
                # 38,40,43,44,45,48,49,54,55,59,60,63,64,67,68,
                # 69,70,71,72,75,78,80,82,87,89,94,95,97
                if fontname == '1' or fontname == '2' or fontname == '6' or fontname == '7' or fontname == '8' \
                or fontname == '9' or fontname == '10' or fontname == '11' or fontname == '12' \
                or fontname == '13' or fontname == '14' or fontname == '20' or fontname == '23' \
                or fontname == '24' or fontname == '28' or fontname == '29' or fontname == '35' \
                or fontname == '38' or fontname == '40' or fontname == '43' or fontname == '44' \
                or fontname == '45' or fontname == '48' or fontname == '49' or fontname == '54' \
                or fontname == '55' or fontname == '59' or fontname == '60' or fontname == '63' \
                or fontname == '64' or fontname == '67' or fontname == '68' or fontname == '69' \
                or fontname == '70' or fontname == '71' or fontname == '72' or fontname == '75' \
                or fontname == '78' or fontname == '80' or fontname == '82' or fontname == '87' \
                or fontname == '89' or fontname == '94' or fontname == '95' or fontname == '97':
                    pass
                
                # 초성,중성,종성이 분리된 경우
                else:
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component = np.vstack([final_component_1, final_component_2])
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    
                    # 중성 컨투어가 두 개인 경우 
                    # 37,42,46,84,96,
                    if fontname == '37' or fontname == '42' or fontname == '46' \
                    or fontname == '84' or fontname == '96':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        middle_component_1 = contours[sorted_contours_indices[2]]
                        middle_component_2 = contours[sorted_contours_indices[3]]
                        
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
                    
                    
                    # 중성 컨투어가 세 개인 경우
                    # 16,81,
                    elif fontname == '16' or fontname == '81':
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        middle_component_1 = contours[sorted_contours_indices[2]]
                        middle_component_2 = contours[sorted_contours_indices[3]]
                        middle_component_3 = contours[sorted_contours_indices[4]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(initial_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_1.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(middle_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(middle_component_2)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(middle_component_3)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_2.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)

                    # 중성 컨투어가 한 개인 경우(그 외)
                    else:
                        initial_component_1 = contours[sorted_contours_indices[0]]
                        initial_component_2 = contours[sorted_contours_indices[1]]
                        initial_component = np.vstack([initial_component_1, initial_component_2])
                        middle_component = contours[sorted_contours_indices[2]]
                        
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
                        
            # 읊, C74A
            elif jamo_dict[j][0]=='ㅇ' and jamo_dict[j][2]=='ㄿ':
                initial_component_1 = contours[sorted_contours_indices[0]]
                initial_component_2 = contours[sorted_contours_indices[1]]
                initial_component = np.vstack([initial_component_1, initial_component_2])
                middle_component = contours[sorted_contours_indices[2]]
                
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
                
                cv2.fillPoly(image_copy,[np.array(initial_component)],(255,255,255))
                cv2.fillPoly(image_copy,[np.array(middle_component)],(255,255,255))
                file_string = f'{fontname}_{char_unicode}_3.png'
                file_path = os.path.join(output_dir, file_string)
                cv2.imwrite(file_path,image_copy)
                
            # 줄, C904
            elif jamo_dict[j][0]=='ㅈ' :
                
                # 초성과 중성이 분리된 경우 
                # 16,17,31,38,39,42,46,50,52,56,74,
                if fontname == '16' or fontname == '17' or fontname == '31' \
                or fontname == '38' or fontname == '39' or fontname == '42' \
                or fontname == '46' or fontname == '50' or fontname == '52' \
                or fontname == '56' or fontname == '74':
                    initial_component = contours[sorted_contours_indices[0]]
                    middle_component = contours[sorted_contours_indices[1]]
                    final_component = contours[sorted_contours_indices[2]]
                    
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
                
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
            # 쪽, CABD
            elif jamo_dict[j][0]=='ㅉ':
            
                # 초성, 중성이 붙어있는 경우
                # 2,3,4,5,6,10,13,15,25,26,33,34,35,38,45,47,48,49,55,56,58,
                # 59,60,63,66,67,68,69,71,72,74,78,81,82,88,90,91,92,94,97
                if fontname == '2' or fontname == '3' or fontname == '4' or fontname == '5' \
                or fontname == '6' or fontname == '10' or fontname == '13' or fontname == '15' \
                or fontname == '25' or fontname == '26' or fontname == '33' or fontname == '34' \
                or fontname == '35' or fontname == '38' or fontname == '45' or fontname == '47' \
                or fontname == '48' or fontname == '49' or fontname == '55' or fontname == '56' \
                or fontname == '58' or fontname == '59' or fontname == '60' or fontname == '63' \
                or fontname == '66' or fontname == '67' or fontname == '68' or fontname == '69' \
                or fontname == '71' or fontname == '72' or fontname == '74' or fontname == '78' \
                or fontname == '81' or fontname == '82' or fontname == '88' or fontname == '90' \
                or fontname == '91' or fontname == '92' or fontname == '94' or fontname == '97':
                    pass
                
                else:
                    final_component = contours[sorted_contours_indices[-1]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
                    middle_component = contours[sorted_contours_indices[-2]]
                    
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
                
            # 틈, D2C8
            elif jamo_dict[j][0]=='ㅌ':
                
                # 종성 ㅁ 컨투어가 한 개인 경우 
                # 7,11,16,28,42,49,56,60,67,68,75,89,
                if fontname == '7' or fontname == '11' or fontname == '16' \
                or fontname == '28' or fontname == '42' or fontname == '49' \
                or fontname == '56' or fontname == '60' or fontname == '67' \
                or fontname == '68' or fontname == '75' or fontname == '89':
                    final_component = contours[sorted_contours_indices[-1]]
                    middle_component = contours[sorted_contours_indices[-2]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
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
                    
                
                # 종성 ㅁ 컨투어가 두 개인 경우 
                # 19,
                elif fontname == '19':
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    middle_component = contours[sorted_contours_indices[-3]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                    cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
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
                    
                
                # 종성과 중성이 붙어있는 경우 
                # 37,
                elif fontname == '37':
                    pass
                
                # 종성 ㅁ 컨투어가 안에 한 개, 밖에 한 개인 경우(그 외)
                else:
                    final_component_1 = contours[sorted_contours_indices[-1]]
                    final_component_2 = contours[sorted_contours_indices[-2]]
                    final_component = np.vstack([final_component_1,final_component_2])
                    middle_component = contours[sorted_contours_indices[-3]]
                    
                    mask = np.zeros((256,256,3), np.uint8) + 255
                    cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                    file_string = f'{fontname}_{char_unicode}_3.png'
                    file_path = os.path.join(output_dir, file_string)
                    cv2.imwrite(file_path,mask)
                    
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

            # 횿, D6BF
            elif jamo_dict[j][0]=='ㅎ':
                
                # 초성, 중성이 붙어있는 경우 
                # 1,2,5,6,7,8,9,10,11,12,13,14,18,19,20,23,24,25,26,29,30,
                # 31,32,35,36,37,38,39,41,43,44,45,47,48,51,53,54,55,56,59
                # 60,63,64,65,67,68,69,70,71,72,73,74,75,76,78,79,80,82,87,88,91,
                # 93,95,96,97,98
                if fontname == '1' or fontname == '2' or fontname == '5' or fontname == '6' \
                or fontname == '7' or fontname == '8' or fontname == '9' or fontname == '10' \
                or fontname == '11' or fontname == '12' or fontname == '13' or fontname == '14' \
                or fontname == '18' or fontname == '19' or fontname == '20' or fontname == '23' \
                or fontname == '24' or fontname == '25' or fontname == '26' or fontname == '29' \
                or fontname == '30' or fontname == '31' or fontname == '32' or fontname == '35' \
                or fontname == '36' or fontname == '37' or fontname == '38' or fontname == '39' \
                or fontname == '41' or fontname == '43' or fontname == '44' or fontname == '45' \
                or fontname == '47' or fontname == '48' or fontname == '51' or fontname == '53' \
                or fontname == '54' or fontname == '55' or fontname == '56' or fontname == '59' \
                or fontname == '60' or fontname == '63' or fontname == '64' or fontname == '65' \
                or fontname == '67' or fontname == '68' or fontname == '69' or fontname == '70' \
                or fontname == '71' or fontname == '72' or fontname == '73' or fontname == '74' \
                or fontname == '75' or fontname == '76' or fontname == '78' or fontname == '79' \
                or fontname == '80' or fontname == '82' or fontname == '87' or fontname == '88' \
                or fontname == '91' or fontname == '93' or fontname == '95' or fontname == '96' \
                or fontname == '97' or fontname == '98':
                    pass
            
                # 초성,중성,종성이 분리된 경우(그 외)
                else:
                    
                    # 종성 ㅊ 컨투어가 한 개인 경우 
                    # 15,49,
                    if fontname == '15' or fontname == '49':
                        final_component = contours[sorted_contours_indices[-1]]
                        middle_component = contours[sorted_contours_indices[-2]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
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
                    
                    # 종성 ㅊ 컨투어가 두 개인 경우
                    # 3,4,17,21,22,23,24,33,34,42,46,50,52,57,62,66,77,81,
                    elif fontname == '3' or fontname == '4' or fontname == '17' or fontname == '21' \
                    or fontname == '22' or fontname == '23' or fontname == '24' or fontname == '33' \
                    or fontname == '34' or fontname == '42' or fontname == '46' or fontname == '50' \
                    or fontname == '52' or fontname == '57' or fontname == '62' or fontname == '66' \
                    or fontname == '77' or fontname == '81':
                        final_component_1 = contours[sorted_contours_indices[-1]]
                        final_component_2 = contours[sorted_contours_indices[-2]]
                        middle_component = contours[sorted_contours_indices[-3]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
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
                        
                    # 종성 ㅊ 컨투어가 세 개인 경우
                    # 16,58,
                    elif fontname == '16' or fontname == '58':
                        final_component_1 = contours[sorted_contours_indices[-1]]
                        final_component_2 = contours[sorted_contours_indices[-2]]
                        final_component_3 = contours[sorted_contours_indices[-3]]
                        middle_component = contours[sorted_contours_indices[-4]]
                        
                        mask = np.zeros((256,256,3), np.uint8) + 255
                        cv2.fillPoly(mask,[np.array(final_component_1)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(final_component_2)],(0,0,0))
                        cv2.fillPoly(mask,[np.array(final_component_3)],(0,0,0))
                        file_string = f'{fontname}_{char_unicode}_3.png'
                        file_path = os.path.join(output_dir, file_string)
                        cv2.imwrite(file_path,mask)
                        
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
                        

            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image-dir', type=str, dest='img_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output-dir', type=str, dest='output_dir', default=DEFAULT_OUTPUT_DIR)
   
    args = parser.parse_args()

    separate_5type(args.img_dir, args.output_dir)