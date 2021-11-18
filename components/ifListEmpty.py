# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app import app
from suanpan.log import logger
from suanpan.app.arguments import String, List


@app.input(List(key="inputData1", default="Suanpan"))
@app.output(List(key="outputData1", alias="out1"))
@app.output(String(key="outputData2", alias="out2"))
def ifListEmpty(context):
    args = context.args
    fish_list = args.inputData1
    if fish_list != []:
        logger.info(f'{fish_list}')
        return {"out1": fish_list}
    else:
        logger.info(f'{fish_list}')
        return {"out2": "cap again"}


if __name__ == "__main__":
    suanpan.run(app)
