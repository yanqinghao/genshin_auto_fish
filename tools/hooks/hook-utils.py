import pathlib
from PyInstaller.utils.hooks import collect_all, collect_system_data_files
from PyInstaller.utils.hooks import logger

datas, binaries, hiddenimports = collect_all('utils', include_py_files=False)

# 如果需要加入其他文件
root = pathlib.Path(__file__).parent.parent.parent
imgs = str(root / 'imgs')
datas += collect_system_data_files(imgs, destdir='imgs')
weights = str(root / 'weights')
datas += collect_system_data_files(weights, destdir='weights')

logger.info('Collecting suanpan components datas: {}'.format(datas))
logger.info('Collecting suanpan components hiddenimports: {}'.format(hiddenimports))