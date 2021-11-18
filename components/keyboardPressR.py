# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import suanpan
import keyboard
from suanpan.app import app
from suanpan.arguments import String
from suanpan.log import logger
from utils import *


restore_saved_module("threading")

def pressGenerator():
    logger.info("Start to detect keyboard R...")
    keyboard.wait('r')
    logger.info("Detect keyboard R Pressed...")
    yield "r"
    logger.info("Send Message to Next Node")
    while True:
        pass


@app.trigger.loop(pressGenerator)
@app.trigger.output(String(key="outputData1"))
def keyboardPressR(_, i):
    return i


if __name__ == "__main__":
    suanpan.run(app)
