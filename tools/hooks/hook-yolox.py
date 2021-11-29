import pathlib
from PyInstaller.utils.hooks import collect_all, collect_system_data_files
from PyInstaller.utils.hooks import logger

datas, binaries, hiddenimports = collect_all('yolox', include_py_files=True)

# # 如果需要加入其他文件
# root = pathlib.Path(__file__).parent.parent.parent
# yolox = str(root / 'yolox')
# datas += collect_system_data_files(yolox, destdir='yolox')

logger.info('Collecting yolox datas: {}'.format(datas))
logger.info('Collecting yolox hiddenimports: {}'.format(hiddenimports))
