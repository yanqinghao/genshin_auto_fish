# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import cv2
import time
import torch
import suanpan
import winsound
from copy import deepcopy
from collections import Counter
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import FISH_CLASSES
from yolox.utils import postprocess, vis

from suanpan.app import app
from suanpan.log import logger
from suanpan.app.arguments import String, Int, Float, Json

from utils import *



@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.param(String(key="param1", alias="demo"))
@app.param(String(key="param2", alias="exp_file"))
@app.param(String(key="param3", alias="ckpt"))
@app.param(Float(key="param4", alias="conf"))
@app.param(Float(key="param5", alias="nms"))
@app.param(Int(key="param6", alias="tsize"))
@app.param(String(key="param7", alias="device"))
@app.output(Json(key="outputData1", alias="msgout"))
def capFish(context):
    args = context.args

    name = None
    experiment_name = None
    fp16 = False
    legacy = False
    fuse = False
    trt = False

    #DQN
    n_states = 3
    n_actions = 2
    step_tick = 12
    model_dir = './weights/fish_genshin_net.pth'

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
    logger.info("Model Summary: {}".format(get_model_info(
        model, exp.test_size)))

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

    predictor = Predictor(model, exp, FISH_CLASSES, trt_file, decoder,
                          args.device, fp16, legacy)

    print('init ok')
    winsound.Beep(500, 500)
    time.sleep(5)
    if args.demo == "image":
        fishlist = get_fish_types(predictor, n=12, rate=0.6)
    logger.info(fishlist)
    winsound.Beep(500, 500)
    return fishlist


#从predictor粘贴的
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

        ratio = min(self.test_size[0] / img.shape[0],
                    self.test_size[1] / img.shape[1])
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
            outputs = postprocess(outputs,
                                  self.num_classes,
                                  self.confthre,
                                  self.nmsthre,
                                  class_agnostic=True)
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
            obj_list.append([
                self.cls_names[int(item[6])], scores,
                [bboxes[0], bboxes[1], bboxes[2], bboxes[3]]
            ])
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


def get_fish_types(predictor, n=12, rate=0.6):
    counter = Counter()
    fx = lambda x: int(np.sign(np.cos(np.pi * (x / (n // 2)) + 1e-4)))
    for i in range(n):
        obj_list = predictor.image_det(cap())
        if obj_list is None:
            mouse_move(70 * fx(i), 0)
            time.sleep(0.1)  #推迟执行的秒数。
            continue
        cls_list = set([x[0] for x in obj_list])
        counter.update(cls_list)
        mouse_move(70 * fx(i), 0)
        time.sleep(0.2)
    fish_list = [k for k, v in dict(counter).items() if v / n >= rate]
    return fish_list


if __name__ == "__main__":
    suanpan.run(app)
