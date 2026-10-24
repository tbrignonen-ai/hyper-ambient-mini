@echo off
rem hyperambient-mini launcher: ensures the STT server is up, then opens the harness.
setlocal
cd /d "%~dp0"

rem Already listening? Then just open the app.
powershell -NoProfile -Command "try { $c = New-Object Net.Sockets.TcpClient('127.0.0.1', 8077); $c.Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 goto :app

echo [hyperambient-mini] starting parakeet-server (CPU, 4 threads, ~400 MB RAM, 0 GPU)...
start "hyperambient-mini-server" /min "bin\parakeet-v0.6.1-bin-win-cpu-x64\parakeet-server.exe" ^
  --model models\fluid-speech-mini-q2k-q4k.gguf --host 127.0.0.1 --port 8077 --threads 4
timeout /t 3 /nobreak >nul

:app
python app\mini_harness.py
endlocal
