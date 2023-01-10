from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import sys
import cv2
import numpy as np
import math
import tool.robot_function as rof
import tool.rw_function as rwf
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode,VisImage
from detectron2.config import get_cfg
import os
import tensorflow
sys.path.append("/content/robo_ga_com")
import tool.robot_function as rof

global iw,ih,ow,oh,PN,MIN_MAX,One_OR_PRO
iw = 200
ih = 200
PN = [100,200,180]
MIN_MAX = [[0,2],[-2,2],[0,math.pi]]
One_or_Pro = 1
  
def making_robot(fuzzy1_paras,fuzzy2_paras):
  robot = rof.obj_func(fuzzy_rule1 = fuzzy1_paras["fuzzy_rule1"],set_types1 = fuzzy1_paras["set_types1"],out_level1 = fuzzy1_paras["out_level1"],
                           fuzzy_rule2 = fuzzy2_paras["fuzzy_rule2"],set_types2 = fuzzy2_paras["set_types2"],out_level2 = fuzzy2_paras["out_level2"],
                           object_items = ["end_time"])
  divi1 =  rwf.output_divi(fuzzy1_paras["fuzzy_rule1"],fuzzy1_paras["set_types1"],fuzzy1_paras["Kch1"],fuzzy1_paras["out_level1"])
  divi2 =  rwf.output_divi( fuzzy2_paras["fuzzy_rule2"], fuzzy2_paras["set_types2"], fuzzy2_paras["Kch2"], fuzzy2_paras["out_level2"])
  divi_list = divi1+divi2
  return robot,divi_list

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

def one_step(robot,time_interval,time_detail,count,maskrcnn_predictor,new_model): #time_intervalは1ステップ何秒であるかを指定する
    #制御周期か否か判断
    obj_point = np.array([[0,0,0],[0,0.865,0]])
    time_judge = ((robot.sum_time*10.0)/(time_interval*10.0)).is_integer()
    if time_judge:
        judge_pose_estimation = 1 ###本来は1
        tmp = robot.pose
        pose_arr = selfpose_function(maskrcnn_predictor,new_model,count)
        if pose_arr.size == 0:
            robot.pose = tmp
        else:
          robot.pose = judge_obj(tmp,obj_point,pose_arr)
          print(robot.pose)
          
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

def predict_mask_rcnn(predictor,img):
  outputs = predictor(img)
  louver_metadata = MetadataCatalog.get("louver")
  v = Visualizer(img[:, :, ::-1],metadata=louver_metadata, scale=1.0,instance_mode=ColorMode.IMAGE_BW)
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
  ###注意############################ masks[0] masks
  if masks.ndim == 2:
    mask_image = img*np.array(masks, dtype=np.uint8).reshape(240,320,-1)
    mask_image = mask_image.astype("uint8")
  elif masks.ndim==3:
    masks_list = []
    for i in range(len(masks)):
      mask_image = img*np.array(masks[i], dtype=np.uint8).reshape(240,320,-1)
      mask_image = mask_image.astype("uint8")
      masks_list.append(mask_image)
    return np.array(masks_list)
  return mask_image

def making_multi_model(filepath):
  new_model = tensorflow.keras.models.load_model(filepath)
  return new_model

def img_change(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.Laplacian(img, cv2.CV_32F, ksize=1)
  img = np.where(img < 0, 0, img)
  img = cv2.resize(img, dsize=(200, 200))
  max_v = np.max(img)
  img = img/max_v
  return img

def result_super_softmax(y_arr,no):
  x2 = np.linspace(MIN_MAX[no][0], MIN_MAX[no][1], PN[no])
  seki = y_arr*x2
  return np.sum(seki)

def pose_calc(predictions):
  c = 0
  pre_li = []
  for pre in predictions:
    for pre_vp in pre:
      vp = result_super_softmax(pre_vp,c)
      pre_li.append(vp)
    c+=1
  return pre_li

def judge_obj(theory_pose,obj_point,pose_arr):
  a = 0.04 
  b = 0.06 
  sum_pose = np.array([0.0,0.0,0.0])
  delta_theory = theory_pose -obj_point
  for i in range(pose_arr.shape[0]):
    index = np.argmin(np.sum((abs(delta_theory - pose_arr[i]))/np.array([2,4,math.pi])*100,axis=1))
    sum_pose +=  obj_point[index]+pose_arr[index]
    pose_non = sum_pose/pose_arr.shape[0]
    pose_true = pose_non-np.array([a*math.cos(pose_non[2])-b*math.sin(pose_non[2]),a*math.sin(pose_non[2])+b*math.cos(pose_non[2]),0])
  return pose_true

def predict_pose(model,mask):
  pose_list = []
  for i in range(len(mask)):
    img_part = img_change(mask[i])
    img_part = img_part.reshape(-1,200,200)
    predictions = model.predict(img_part)
    pose_list.append(pose_calc(predictions))
  pose_arr =  np.array(pose_list)
  if pose_arr.size==0:
    return pose_arr
  else:
    return pose_arr

def selfpose_function(maskrcnn_predictor,new_model,number):
  ##画像取得
  img_path = take_photo(filename='/content/drive/MyDrive/exp_image_folder/img.jpg', quality=0.8,No=number)
  img = cv2.imread(img_path)
  img = cv2.resize(img,(320,240))
  #img = cv2.imread("/content/drive/MyDrive/image_sample/image_20220913_1_0.png")
  #MaskRCNN
  mask= predict_mask_rcnn(maskrcnn_predictor,img)
  #マルチタスク位置推定
  pose_arr = predict_pose(new_model,mask)
  return pose_arr
