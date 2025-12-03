@echo off
echo Setting up Python environment for HardCoReNAS...

python -m venv venv
call venv\Scripts\activate

echo Installing dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r Docker/requirements.txt

echo Setup complete. Activate venv with: venv\Scripts\activate
pause
