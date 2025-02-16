python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python requirements.py
.venv\Scripts\python setup.py develop
.venv\Scripts\python -m pip install pyinstaller

.venv\Scripts\pyinstaller --additional-hooks-dir tools/hooks --clean --noconfirm --path "D:\code\genshin_auto_fish-1\.venv\Lib\site-packages\cv2" -D run.py -n fishing

Compress-Archive "./dist/*" -DestinationPath 'auto-fishing.zip'
