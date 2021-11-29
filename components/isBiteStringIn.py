# -*- coding: utf-8 -*-
#from __future__ import absolute_import, print_function

import cv2
import time
import winsound
import numpy as np
from copy import deepcopy

import suanpan
from suanpan.app import app
from suanpan.log import logger
from suanpan.app.arguments import String, Json

from utils import *


@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.output(String(key="outputData1", alias="out1"))
@app.output(String(key="outputData2", alias="out2"))
def ifBite(context):
    args = context.args
    logger.info(args)
    #fishlist = args.inputData1

    winsound.Beep(330, 550)
    #time.sleep(1)

    bite_timeout = 25
    times = 0
    env = Fishing(delay=0.1, max_step=10000, show_det=True)
    while True:
        if env.is_bite():
            env.drag()
            return "bite", None
            #return {"out1":"bite"}

        #if env.is_bite()==False and times < bite_timeout and 

        time.sleep(0.5)
        times += 1
        if times > bite_timeout:
            env.drag()
            time.sleep(1)  
            return None, "No bite"
            #return {"out2":fishlist}


class Fishing:
    def __init__(self,
                 delay=0.1,
                 max_step=100,
                 show_det=True):  #''', predictor=None'''
        self.t_l = cv2.imread('imgs/target_left.png')
        self.t_r = cv2.imread('imgs/target_right.png')
        self.t_n = cv2.imread('imgs/target_now.png')
        self.im_bar = cv2.imread('imgs/bar2.png')
        self.bite = cv2.imread('imgs/bite.png', cv2.IMREAD_GRAYSCALE)
        self.std_color = np.array([192, 255, 255])
        self.r_ring = 21
        self.delay = delay
        self.max_step = max_step
        self.count = 0
        self.show_det = show_det

        self.add_vec = [0, 2, 0, 2, 0, 2]

    def reset(self):
        self.y_start = self.find_bar()[0]
        self.img = cap([712 - 10, self.y_start, 496 + 20, 103])

        self.fish_start = False
        self.zero_count = 0
        self.step_count = 0
        self.reward = 0
        self.last_score = self.get_score()

        return self.get_state()

    def drag(self):
        mouse_click(1630, 995)  #短暂点击鼠标开始拉杆？为啥要移到这个位置

    def do_action(self, action):
        if action == 1:
            self.drag()

    def scale(self, x):
        return (x - 5 - 10) / 484

    def find_bar(self, img=None):
        img = cap(region=[700, 0, 520, 300]) if img is None else img[:300,
                                                                     700:700 +
                                                                     520, :]
        bbox_bar = match_img(img, self.im_bar)
        if self.show_det:
            img = deepcopy(img)
            cv2.rectangle(img, bbox_bar[:2], bbox_bar[2:4], (0, 0, 255),
                          1)  # 画出矩形位置
            cv2.imwrite(f'img_tmp/bar.jpg', img)
        return bbox_bar[1] - 9, bbox_bar

    def is_bite(self):
        img = cap(region=[1595, 955, 74, 74])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_output = cv2.Canny(gray, 50, 150)  #进行图像边缘检测
        #logger.info(f'{self.bite} and {edge_output}')
        #logger.info(psnr(self.bite, edge_output))
        return psnr(self.bite, edge_output) > 10

    def get_state(self, all_box=False):
        bar_img = self.img[2:34, :, :]
        bbox_l = match_img(bar_img, self.t_l)
        bbox_r = match_img(bar_img, self.t_r)
        bbox_n = match_img(bar_img, self.t_n)

        bbox_l = tuple(list_add(bbox_l, self.add_vec))
        bbox_r = tuple(list_add(bbox_r, self.add_vec))
        bbox_n = tuple(list_add(bbox_n, self.add_vec))

        if self.show_det:
            img = deepcopy(self.img)
            cv2.rectangle(img, bbox_l[:2], bbox_l[2:4], (255, 0, 0),
                          1)  # 画出矩形位置
            cv2.rectangle(img, bbox_r[:2], bbox_r[2:4], (0, 255, 0),
                          1)  # 画出矩形位置
            cv2.rectangle(img, bbox_n[:2], bbox_n[2:4], (0, 0, 255),
                          1)  # 画出矩形位置
            fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
            fontScale = 1
            thickness = 1
            cv2.putText(img,
                        str(self.last_score), (257 + 30, 72),
                        fontScale=fontScale,
                        fontFace=fontFace,
                        thickness=thickness,
                        color=(0, 255, 255))
            cv2.putText(img,
                        str(self.reward), (257 + 30, 87),
                        fontScale=fontScale,
                        fontFace=fontFace,
                        thickness=thickness,
                        color=(255, 255, 0))
            cv2.imwrite(f'img_tmp/{self.count}.jpg', img)
        self.count += 1

        #voc dataset
        '''cv2.imwrite(f'./bar_dataset/{self.count}.jpg', self.img)
        with open(f'./bar_dataset/{self.count}.xml', 'w', encoding='utf-8') as f:
            f.write(self.voc_tmp.format(self.count, *bbox_l[:4], *bbox_r[:4], *bbox_n[:4]))'''
        if all_box:
            return bbox_l, bbox_r, bbox_n
        else:
            return self.scale(bbox_l[4]), self.scale(bbox_r[4]), self.scale(
                bbox_n[4])

    def get_score(self):
        cx, cy = 247 + 10, 72
        for x in range(4, 360, 2):
            px = int(cx + self.r_ring * np.sin(np.deg2rad(x)))
            py = int(cy - self.r_ring * np.cos(np.deg2rad(x)))
            if np.mean(np.abs(self.img[py, px, :] - self.std_color)) > 50:
                return x // 2 - 2
        return 360 // 2 - 2

    def step(self, action):
        self.do_action(action)

        time.sleep(self.delay - 0.05)
        self.img = cap([712 - 10, self.y_start, 496 + 20, 103])
        self.step_count += 1

        score = self.get_score()
        if score > 0:
            self.fish_start = True
            self.zero_count = 0
        else:
            self.zero_count += 1
        self.reward = score - self.last_score
        self.last_score = score

        return self.get_state(), self.reward, (self.step_count > self.max_step
                                               or (self.zero_count >= 6
                                                   and self.fish_start)
                                               or score > 176)

    def render(self):
        pass


if __name__ == "__main__":
    suanpan.run(app)
