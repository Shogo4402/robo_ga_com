from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
import math
import tool.robot_function as rof
import tool.rw_function as rwf
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
import os

def take_photo(filename='img.jpg', quality=0.8,No=0):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  names =filename.split(".")
  names.insert(1,str(No)+".")
  filename_w = "".join(names)

  with open(filename_w, 'wb') as f:
    f.write(binary)
  return filename_w

def one_step(robot,time_interval,time_detail): #time_intervalは1ステップ何秒であるかを指定する
    #制御周期か否か判断
    time_judge = ((robot.sum_time*10.0)/(time_interval*10.0)).is_integer()
    if time_judge:
        judge_pose_estimation = 0 ###本来は1
        nu,omega=robot.decision(robot.pose)
        robot.nu_m,robot.omega = nu,omega #速度・角速度更新
    else:
        judge_pose_estimation = 0
        nu,omega= robot.nu_m,robot.omega
    robot.sum_time = (robot.sum_time*10+(10*time_detail))/10 #合計時間
    
    #pose記録
    robot.li1.append(robot.pose[0])
    robot.li2.append(robot.pose[1])
    robot.li3.append(robot.pose[2])
    robot.now_nos.append(robot.now_no)
    #time_detail秒後の自己位置計算
    if judge_pose_estimation:
        #位置更新
        pass
    else:
        robot.pose=robot.state_transition(robot.nu_m,omega,robot.time_detail,robot.pose)
        
    if robot.pose[2] >= 2*math.pi:
        robot.pose[2]=robot.pose[2] - 2*math.pi
    elif robot.pose[2] < 0:
        robot.pose[2] = robot.pose[2] +2*math.pi
    #目標切り替えのための軌跡登録
    robot.orbit_register(robot.pose)
    return robot.move_end,judge_pose_estimation

#Mask CNNのpredictor生成
def making_maskrcnn_predictor(filepath):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1クラスのみ
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, filepath)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9 ##kakuritu
  #cfg.MODEL.DEVICE = "cpu"
  predictor = DefaultPredictor(cfg)
  return predictor
