@echo off
setlocal

echo --- HardCoReNAS Windows Deployment Demo ---

:: 1. Check Environment
if not exist "env\Scripts\activate.bat" (
    echo [!] Virtual environment not found. Please run scripts\setup_env.bat first.
    pause
    exit /b 1
)

call env\Scripts\activate.bat

:: 2. Check Model
if not exist "model_quantized.onnx" (
    echo [!] Quantized model not found. Please run scripts\run_nas_pipeline.py first.
    pause
    exit /b 1
)

:: 3. Run Benchmark
echo.
echo [1/2] Running Benchmark (Latency & Size)...
python benchmark/evaluate.py --model_path model_quantized.onnx

:: 4. Run Inference Test
echo.
echo [2/2] Running Inference Test...
python test_inference.py

echo.
echo --- Demo Completed Successfully ---
pause
