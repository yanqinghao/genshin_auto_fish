import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Image, Npy, Json, List,Float, Int
from suanpan.log import logger

import argparse
import os
import time

#from loguru import logger

import torch
import keyboard
import winsound

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info

#from fisher.environment import *
#from fisher.predictor import *
#from fisher.models import FishNet

import time
import cv2

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import FISH_CLASSES
from yolox.utils import postprocess, vis

from utils import *
import pyautogui
from copy import deepcopy
from collections import Counter
import traceback


#@app.input(Image(key="inputData1", default="Suanpan"))
@app.input(List(key="inputData1", alias="msgin",default="Suanpan"))
#@app.input(String(key="inputData1", alias="msgin",default="Suanpan"))
#@app.param(String(key="param1", alias="windowname"))
#@app.param(String(key="param2", alias="region"))
@app.param(String(key="param1", alias="demo"))
@app.param(String(key="param2", alias="exp_file"))
@app.param(String(key="param3", alias="ckpt"))
@app.param(Float(key="param4", alias="conf"))
@app.param(Float(key="param5", alias="nms"))
@app.param(Int(key="param6", alias="tsize"))
@app.param(String(key="param7", alias="device"))
@app.output(List(key="outputData1",alias="msgout"))

def hello_world(context):
    args = context.args
    #cap=args.inputData1
    #args.exp_file='yolox/exp/yolox_tiny_fish.py' 右栏
    #trt_file='weights/best_tiny3.pth' #右栏
    #ckpt='weights/best_tiny3.pth' #右栏
    #device='cpu' #右栏
    name=None
    experiment_name=None
    fp16=False
    legacy=False
    fuse=False
    trt=False

    #DQN
    n_states=3
    n_actions=2
    step_tick=12
    model_dir='./components/weights/fish_genshin_net.pth'

    exp = get_exp(args.exp_file, name)

    if not experiment_name:
        experiment_name = exp.exp_name

    if trt:
        args.device = "gpu"

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))


    if args.device == "gpu":
        model.cuda()
        if fp16:
            model.half()  # to FP16
    model.eval()


    if not trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if trt:
        assert not fuse, "TensorRT model is not support model fusing!"
        if args.ckpt is None:
            trt_file = os.path.join(file_name, "model_trt.pth")
        else:
            trt_file = args.ckpt
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, FISH_CLASSES, trt_file, decoder, args.device, fp16, legacy)

    #agent = FishNet(in_ch=n_states, out_ch=n_actions) #DQN
    #agent.load_state_dict(torch.load(model_dir))
    #agent.eval()

    fishlist = args.inputData1
    #typestr= args.inputData1 #前接输入框输['hua jiang']
    #fishtype=eval(typestr)
    fish_type=fishlist[0]
    #cap()
    winsound.Beep(330, 550)
    time.sleep(8)
    throw_rod(predictor,fish_type)
    logger.info(f'{fish_type}')
    return fishlist


    
    
def throw_rod(predictor,fish_type):
    food_imgs = [
            cv2.imread('./imgs/food_gn.png'),
            cv2.imread('./imgs/food_cm.png'),
            cv2.imread('./imgs/food_bug.png'),
            cv2.imread('./imgs/food_fy.png'),
        ]
    
    ff_dict={'hua jiang':0, 'ji yu':1, 'die yu':2, 'jia long':3, 'pao yu':3}
    dist_dict={'hua jiang':130, 'ji yu':80, 'die yu':80, 'jia long':80, 'pao yu':80}
    food_rgn=[580,400,740,220]
    last_fish_type='hua jiang'
    #self.last_fish_type='die yu' # 钓雷鸣仙
    show_det=True #show detail: true or false
    os.makedirs('img_tmp/', exist_ok=True)    

    mouse_down(960, 540)
    winsound.Beep(800, 400)
    time.sleep(1)

    def move_func(dist):
        if dist>100:
            return 50 * np.sign(dist)
        else:
            return (abs(dist)/2.5+10) * np.sign(dist)

    for i in range(50):
        try:
            obj_list, outputs, img_info = predictor.image_det(cap(), with_info=True)
            if show_det:
                cv2.imwrite(f'img_tmp/det{i}.png', predictor.visual(outputs[0],img_info))
                #实际上没有读取这些png
            rod_info = sorted(list(filter(lambda x: x[0] == 'rod', obj_list)), key=lambda x: x[1], reverse=True)
            if len(rod_info)<=0:
                mouse_move(np.random.randint(-50,50), np.random.randint(-50,50))
                time.sleep(0.1)
                continue
            rod_info=rod_info[0]
            rod_cx = (rod_info[2][0] + rod_info[2][2]) / 2
            rod_cy = (rod_info[2][1] + rod_info[2][3]) / 2

            fish_info = min(list(filter(lambda x: x[0] == fish_type, obj_list)),
                            key=lambda x: distance((x[2][0]+x[2][2])/2, (x[2][1]+x[2][3])/2, rod_cx, rod_cy))

            if (fish_info[2][0] + fish_info[2][2]) > (rod_info[2][0] + rod_info[2][2]):
                #dist = -self.dist_dict[fish_type] * np.sign(fish_info[2][2] - (rod_info[2][0] + rod_info[2][2]) / 2)
                x_dist = fish_info[2][0] - dist_dict[fish_type] - rod_cx
            else:
                x_dist = fish_info[2][2] + dist_dict[fish_type] - rod_cx

            print(x_dist, (fish_info[2][3] + fish_info[2][1]) / 2 - rod_info[2][3])
            if abs(x_dist)<30 and abs((fish_info[2][3] + fish_info[2][1]) / 2 - rod_info[2][3])<30:
                break

            dx = int(move_func(x_dist))
            dy = int(move_func(((fish_info[2][3]) + fish_info[2][1]) / 2 - rod_info[2][3]))
            mouse_move(dx, dy)
        except Exception as e:
            traceback.print_exc()
        #time.sleep(0.3)
    mouse_up(960, 540)
    winsound.Beep(800, 400)


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=FISH_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def image_det(self, img, with_info=False):
        outputs, img_info = self.inference(img)
        ratio = img_info["ratio"]
        obj_list = []
        if outputs[0] is None:
            return None
        for item in outputs[0].cpu():
            bboxes = item[:4]
            # preprocessing: resize
            bboxes /= ratio
            scores = item[4] * item[5]
            obj_list.append([self.cls_names[int(item[6])], scores, [bboxes[0], bboxes[1], bboxes[2], bboxes[3]]])
        if with_info:
            return obj_list, outputs, img_info
        else:
            return obj_list


    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

if __name__ == "__main__":
    suanpan.run(app)
