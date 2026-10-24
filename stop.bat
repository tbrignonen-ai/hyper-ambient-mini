@echo off
rem Stop the hyperambient-mini app and its STT server, cleanly.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*mini_harness*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
powershell -NoProfile -Command "Get-Process parakeet-server -ErrorAction SilentlyContinue | Stop-Process -Force"
echo hyperambient-mini stopped (app + server).
timeout /t 2 /nobreak >nul
