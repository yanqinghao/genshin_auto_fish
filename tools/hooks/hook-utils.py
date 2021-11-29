import pathlib
from PyInstaller.utils.hooks import collect_all, collect_system_data_files
from PyInstaller.utils.hooks import logger

datas, binaries, hiddenimports = collect_all('utils', include_py_files=False)



logger.info('Collecting utils datas: {}'.format(datas))
logger.info('Collecting utils hiddenimports: {}'.format(hiddenimports))