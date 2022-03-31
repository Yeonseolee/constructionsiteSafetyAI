# 구글 코랩에서 실행

# %mkdir /content/constructionSite_safety/
# %cd /content/constructionSite_safety/


# 라벨링 처리된 이미지 데이터 불러오기(데이터 전처리)
# !curl -L 'https://app.roboflow.com/ds/gnJdPPJlvc?key=lAUPjtDa4W' > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# Yolo v5 깃허브 다운로드
# %cd /content
# !git clone https://github.com/ultralytics/yolov5.git

# 설정파일 설치
# %cd /content/yolov5/
# !pip install -r requirements.txt


# 1. Yolo v5 데이터셋 생성
import yaml

with open('/content/constructionSite_safety/data.yaml', 'r') as f:
  data = yaml.full_load(f)

print(data)

data['train'] = '/content/constructionSite_safety/'
data['test'] = '/content/constructionSite_safety/'
data['val'] = '/content/constructionSite_safety/'

with open('/content/constructionSite_safety/data.yaml', 'w') as f:
  yaml.dump(data, f)

print(data)

# 2. Yolov5 학습 : 둘 중에 하나 사용할 것

# 2-1.
# !python train.py --img 416 --batch 16 --epochs 50 --data /content/yolov5_mask/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name mask_yolov5s_results


# 2-2.
# %cd /content/yolov5/
# !python train.py --batch 16 --epochs 50 --data /content/constructionSite_safety/data.yaml --cfg ./models/yolov5s.yaml 
#   --weights yolov5s.pt --name safety_yolov5s_results


# 3. tensorBoard를 이용한 성능평가
# %load_ext tensorboard
# %tensorboard --logdir /content/yolov5/runs/



# 굴착기, 트럭과 사람과의 안전거리를 위해 yolov5 라이브러리 내 좌표값 활용 코드 변경 부분
# detec.py 151번줄 변경

# 코드 변경 부분 
                # project_label = [] # project에 사용할 값들 담아놓는 리스트. label
                # project_location = [] # 좌표값
                # # Write results
                # for *xyxy, conf, cls in reversed(det):
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 좌표값 사용가능하도록 전환
                #     project_label.append([names[int(cls), xywh]]) # 각 객체의 [label, xywh] 저장
                #     if save_txt:  # Write to file
                #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                #         with open(txt_path + '.txt', 'a') as f:
                #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                #     if save_img or save_crop or view_img:  # Add bbox to image
                #         c = int(cls)  # integer class
                #         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                #         annotator.box_label(xyxy, label, color=colors(c, True))
                #         if save_crop:
                #             save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                #     p = project_label.index('person')
                #     d = project_label.index('dump truck')
                #     e = project_label.index('Excavator')
                #     if 'worker' and ('dump truck' or 'Excavator') in project_label:
                #       if project_location[p][0] < project_location[d][0] or project_location[p][0] < project_location[e][0]:
                #         if project_location[p][0] + project_location[p][2] - project_location[d][0] < 10:
                #           print('dump is too close')
                #         elif project_location[p][0] + project_location[p][2] - project_location[e][0] < 10:
                #           print('excavator is too close')





# 4-1. 이미지 데이터로 학습결과 확인1
# %cd /content/yolov5/

from glob import glob

img_list = glob('/content/constructionSite_safety/test/images/*.jpg')

print(len(img_list))

from IPython.display import Image
import os

val_img_path = img_list

weights_path = '/content/yolov5/runs/train/safety_yolov5s_results/weights/best.pt'

# !python detect.py --weights "{weights_path}" --img 416 --conf 0.5 --source "{val_img_path}" 

detect_img_path = '/content/yolov5/runs/detect/exp'




# 4-2. 이미지로 테스트 2
# !python detect.py --source '/content/constructionSite_safety/test/images/*.jpg' --weights '/content/yolov5/runs/train/safety_yolov5s_results/weights/best.pt' --conf 0.3 



# 4-3 이미지로 테스트 3: torch 사용
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print() 








