# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import keyboard
import suanpan
from suanpan.app import app
from suanpan.log import logger
from suanpan.app.arguments import String
import winsound

@app.input(String(key="inputData1", alias="msgin", default="Suanpan"))
@app.output(String(key="outputData1", alias="out1"))
def pressGenerator(context):
    args=context.args
    logger.info(args)
    logger.info("Start to detect keyboard R...")
    winsound.Beep(500,500)
    keyboard.wait('r')
    logger.info("Detect keyboard R Pressed...")
    #app.sendWithoutContext("r")
    logger.info("Send Message to Next Node")
    return 'r'

if __name__ == "__main__":
    suanpan.run(app)
    