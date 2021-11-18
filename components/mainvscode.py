import suanpan
from suanpan.app import app
from suanpan.app.arguments import String, Image, Npy, Json
#from suanpan.log import logger
import time
import cv2
import pyautogui
import numpy as np
import win32api, win32con, win32gui, win32ui
from pathlib import Path
import yaml

#@app.input(Image(key="inputData1", default="Suanpan"))
@app.input(String(key="inputData1", alias="msgin",default="Suanpan"))
#@app.param(String(key="prefix", alias="prefix"))
@app.output(Json(key="outputData1",alias="msgout"))

def hello_suanpan(context):
    #CONFIG_PATH = Path(__file__).parent.parent.joinpath("config.yaml")
    #assert CONFIG_PATH.is_file()

    with open(r'C:/Users/Administrator/Desktop/windows-component/config.yaml', encoding='utf-8') as f:
        result = yaml.safe_load(f)
        DEFAULT_MONITOR_WIDTH = result.get("windows").get("monitor_width")
        DEFAULT_MONITOR_HEIGHT = result.get("windows").get("monitor_height")
        WINDOW_NAME = result.get("game").get("window_name")

    handle = win32gui.FindWindow(None, WINDOW_NAME)   

    win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0) #restore窗口 
    hwnd=win32gui.FindWindow(None, WINDOW_NAME)
    
    args = context.args #前面一个组件传region大小过来, region=[1595, 955, 74, 74]
    regionstr= args.inputData1 
    region=eval(regionstr)

    if region is not None:
        left, top, w, h = region
        # w = x2 - left + 1
        # h = y2 - top + 1
    else:
        w = DEFAULT_MONITOR_WIDTH  # set this
        h = DEFAULT_MONITOR_HEIGHT  # set this
        left = 0
        top = 0

    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC() #创建内存设备描述表
    dataBitMap = win32ui.CreateBitmap() #创建位图对象准备保存图片

    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)

    cDC.SelectObject(dataBitMap) #将截图保存到saveBitMap中
    cDC.BitBlt((0, 0), (w, h), dcObj, (left, top), win32con.SRCCOPY) #保存bitmap到内存设备描述表
    # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
    signedIntsArray = dataBitMap.GetBitmapBits(True) #获取位图信息
    #img = np.fromstring(signedIntsArray, dtype="uint8")#img data array
    img = np.frombuffer(signedIntsArray, dtype="uint8")
    img.shape = (h, w, 4)
    cvtcolor=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) #颜色空间转换,cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return cvtcolor.tolist() 

if __name__ == "__main__":
    suanpan.run(app)
