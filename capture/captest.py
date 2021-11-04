import time
import cv2
import pyautogui
import numpy as np
import win32api, win32con, win32gui, win32ui
from pathlib import Path
import yaml
'''
def get_window_pos(name):
    name = name
    handle = win32gui.FindWindow(None, name)
 # 获取窗口句柄
    if handle == 0:
        return None
    else:
        return win32gui.GetWindowRect(handle), handle

(x1, y1, x2, y2), handle = get_window_pos('原神')
print(x1,y1,x2,y2)

'''
CONFIG_PATH = Path(__file__).parent.parent.joinpath("config.yaml")
assert CONFIG_PATH.is_file()

with open(CONFIG_PATH, encoding='utf-8') as f:
    result = yaml.safe_load(f)
    DEFAULT_MONITOR_WIDTH = result.get("windows").get("monitor_width")
    DEFAULT_MONITOR_HEIGHT = result.get("windows").get("monitor_height")
    WINDOW_NAME = result.get("game").get("window_name")
handle = win32gui.FindWindow(None, WINDOW_NAME)   

win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0) #restore窗口   
#win32gui.SendMessage(handle, win32con.WM_SYSCOMMAND, win32con.SC_MAXIMIZE, 0) #最大化窗口
#win32gui.SetForegroundWindow(handle)# 设为高亮

hwnd=win32gui.FindWindow(None, WINDOW_NAME)

#(x1, y1, x2, y2)=win32gui.GetWindowRect(hwnd)
#print(x1,y1,x2,y2)

#hwnd=win32gui.FindWindow(None, '原神')

left, top, right, bottom = win32gui.GetWindowRect(hwnd)
w=right #windows(monitor)_width
h=bottom #DEFAULT_MONITOR_HEIGHT
print(left, top, right, bottom)

'''
region=None
if region is not None:
    left, top, w, h = region
    # w = x2 - left + 1
    # h = y2 - top + 1
else:
    w = DEFAULT_MONITOR_WIDTH  # set this
    h = DEFAULT_MONITOR_HEIGHT  # set this
    left = 0
    top = 0
'''

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
cvtcolor=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
# Free Resources
dcObj.DeleteDC()
cDC.DeleteDC()
win32gui.ReleaseDC(hwnd, wDC)
win32gui.DeleteObject(dataBitMap.GetHandle())

print(cvtcolor)
filename='capquanping4.png'
cv2.imwrite(filename, cvtcolor, [int(cv2.IMWRITE_PNG_COMPRESSION), 1]) 
#cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) #颜色空间转换,cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
#cv2.cvtColor输出是'numpy.ndarray'
