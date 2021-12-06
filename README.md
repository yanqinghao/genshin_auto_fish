# Introduction
原神自动钓鱼AI由[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), DQN两部分模型组成。使用迁移学习，半监督学习进行训练。
模型也包含一些使用opencv等传统数字图像处理方法实现的不可学习部分。

其中YOLOX用于鱼的定位和类型的识别以及鱼竿落点的定位。DQN用于自适应控制钓鱼过程的点击，让力度落在最佳区域内。

# 流程图
<font color=#66CCFF>**长流程图**</font>

# 模板和组件包下载
长流程全自动模板下载链接：https://suanpan-public.oss-cn-shanghai.aliyuncs.com/suanpan-genshin-fishing/%E9%92%93%E9%B1%BC-%E5%85%A8%E8%87%AA%E5%8A%A8

短流程半自动模板下载链接：https://suanpan-public.oss-cn-shanghai.aliyuncs.com/suanpan-genshin-fishing/%E9%92%93%E9%B1%BC-%E5%8D%8A%E8%87%AA%E5%8A%A8

组件包下载：https://suanpan-public.oss-cn-shanghai.aliyuncs.com/suanpan-genshin-fishing/fishing.zip

# 使用视频



# 流程说明
<font color=#66CCFF>**键盘R触发下游组件、键盘R继续钓鱼**</font>

按r键触发下一个组件

<font color=#66CCFF>**采集图片，目标检测**</font>

在右侧参数设置栏输入yolox所需的对应参数，开始运行后print('init ok'),电脑发出beep声。等待2秒后，游戏人物开始张望（代码控制鼠标小幅移动），采集游戏截图并检测图中是否有鱼和其种类，打印日志并输出捕捉到的鱼种类list，再次响起beep声表示改组件运行完毕。

<font color=#66CCFF>**判断是否捕捉到鱼**</font>

判断前组件传来的list是否为[]。如果为[]，则重新运行“采集图片，目标检测”；如果不为[]，则输出上个组件传来的fish list给下一个组件。

<font color=#66CCFF>**抛竿钓鱼**</font>

在右侧参数设置栏输入yolox所需的对应参数，接收到fish list，发出较低的beep声，等待1秒后代码控制鼠标按下（不松开），发出较高beep声，采集游戏截图并分析鱼饵和鱼之间的距离，根据鱼的位置移动鼠标，到达理想位置后代码控制鼠标松开，发出较高beep声，输出接收到的fish list。

<font color=#66CCFF>**是否上钩**</font>

接收fish list（长流程图），发出较低beep声，判断游戏界面右下角的上钩标志是否改变。如果15秒内没有改变，则输出fish list给“抛竿钓鱼”组件重新抛竿；如果15秒内改变，则输出字符串“bite”触发下一个组件。

接收string（短流程图），发出较低beep声，判断游戏界面右下角的上钩标志是否改变。如果15秒内没有改变，则输出no bite给“键盘R继续钓鱼”组件启动是否上钩监控；如果15秒内改变，则输出字符串“bite”触发下一个组件。

<font color=#66CCFF>**强化学习拉杆**</font>

在右侧参数设置栏输入强化学习预测所需参数，运行后发出较低beep声，代码控制鼠标按键，使力度小块保持在力度条中。结束后输出字符串“next round”，从“采集图片，目标检测”组件（长流程图）或“键盘R继续钓鱼”组件（短流程图）进行下一轮。
