# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import shutil
import keyboard
from suanpan.app import app
from suanpan.log import logger


def pressGenerator():
    if os.path.exists("img_tmp"):
        shutil.rmtree("img_tmp")
    logger.info("Start to detect keyboard R...")
    keyboard.wait('r')
    logger.info("Detect keyboard R Pressed...")
    app.sendWithoutContext("r")
    logger.info("Send Message to Next Node")
    while True:
        pass


if __name__ == "__main__":
    pressGenerator()
