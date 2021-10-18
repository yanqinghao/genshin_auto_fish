python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python requirements.py
.venv\Scripts\python setup.py develop
.venv\Scripts\python -m pip install pyinstaller

.venv\Scripts\pyinstaller --additional-hooks-dir tools/hooks --clean --noconfirm -D run.py -n suanpan-data-analysis

Compress-Archive "./dist/*" -DestinationPath 'auto-fishing.zip'
