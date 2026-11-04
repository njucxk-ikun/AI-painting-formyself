@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================
echo   AI绘画工作台 v2.0 启动器
echo ========================================
echo.
echo 正在启动Streamlit应用...
echo.
python -m streamlit run app.py
pause
