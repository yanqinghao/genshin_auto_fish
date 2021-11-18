# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import gevent
import suanpan
import keyboard
from suanpan.app import app
from suanpan.arguments import String
from suanpan.log import logger


def restore_saved_module(module):
    """
    gevent monkey patch keeps a list of all patched modules.
    This will restore the original ones
    :param module: to unpatch
    :return:
    """
    # Check the saved attributes in geven monkey patch
    if not (module in gevent.monkey.saved):
        return
    _module = __import__(module)

    # If it exist unpatch it
    for attr in gevent.monkey.saved[module]:
        if hasattr(_module, attr):
            setattr(_module, attr, gevent.monkey.saved[module][attr])


def pressGenerator():
    restore_saved_module("threading")
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
