@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: ═══════════════════════════════════════════════════════════════════════
:: build.bat — LWF v2.0 一键打包脚本
::
:: 使用方式：双击运行，或在命令行执行：build.bat
::
:: 前置要求：
::   1. Python 3.9+ 已安装并在 PATH 中
::   2. pip install pyinstaller faster-whisper customtkinter onnxruntime
::   3. bin/ 目录下已放好 ffmpeg.exe（和可选的 CUDA dll）
:: ═══════════════════════════════════════════════════════════════════════

echo.
echo ╔══════════════════════════════════════╗
echo ║   LWF v2.0 Build Script            ║
echo ║   Lite-Whisper-Faster 打包工具      ║
echo ╚══════════════════════════════════════╝
echo.

:: ── 定位项目目录 ──
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "SPEC_FILE=%PROJECT_DIR%LWF.spec"
set "DIST_DIR=%PROJECT_DIR%dist\LWF"
set "BUILD_DIR=%PROJECT_DIR%build"

:: ── Step 1: 检查环境 ──
echo [1/5] 检查环境...

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python！请确保 Python 3.9+ 已安装并在 PATH 中。
    pause
    exit /b 1
)

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 PyInstaller！请运行: pip install pyinstaller
    pause
    exit /b 1
)

python -c "import faster_whisper" >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 faster_whisper！请运行: pip install faster-whisper
    pause
    exit /b 1
)

python -c "import customtkinter" >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 customtkinter！请运行: pip install customtkinter
    pause
    exit /b 1
)

echo ✅ Python 环境就绪

:: ── Step 2: 清理旧构建 ──
echo.
echo [2/5] 清理旧构建...

if exist "%DIST_DIR%" (
    rmdir /s /q "%DIST_DIR%"
    echo    已删除 dist/LWF/
)
if exist "%BUILD_DIR%" (
    rmdir /s /q "%BUILD_DIR%"
    echo    已删除 build/
)

echo ✅ 清理完成

:: ── Step 3: 执行 PyInstaller ──
echo.
echo [3/5] 执行 PyInstaller 打包...
echo    （这可能需要 1-5 分钟，请耐心等待）
echo.

pyinstaller "%SPEC_FILE%" --noconfirm --clean
if errorlevel 1 (
    echo.
    echo ❌ PyInstaller 打包失败！请检查上方错误信息。
    pause
    exit /b 1
)

echo.
echo ✅ PyInstaller 打包完成

:: ── Step 4: 复制资源目录 ──
echo.
echo [4/5] 复制资源目录...

:: 复制 bin/
if exist "%PROJECT_DIR%bin" (
    echo    复制 bin/ → dist/LWF/bin/
    xcopy "%PROJECT_DIR%bin" "%DIST_DIR%\bin\" /E /I /Y /Q >nul
    echo    ✅ bin/ 复制完成
) else (
    echo    ⚠ bin/ 目录不存在，创建空目录
    mkdir "%DIST_DIR%\bin" 2>nul
)

:: 复制 models/
if exist "%PROJECT_DIR%models" (
    echo    复制 models/ → dist/LWF/models/
    xcopy "%PROJECT_DIR%models" "%DIST_DIR%\models\" /E /I /Y /Q >nul
    echo    ✅ models/ 复制完成
) else (
    echo    创建空 models/ 目录
    mkdir "%DIST_DIR%\models" 2>nul
)

:: 确保 temp/ 目录存在
mkdir "%DIST_DIR%\temp" 2>nul

:: ── Step 5: 汇总 ──
echo.
echo [5/5] 打包完成！
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  输出目录: dist\LWF\                                   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

:: 列出关键文件
echo    目录结构:
echo    dist/LWF/
echo    ├── LWF.exe
if exist "%DIST_DIR%\bin" (
    echo    ├── bin/
    for %%f in ("%DIST_DIR%\bin\*") do echo    │   ├── %%~nxf
)
echo    ├── models/
echo    ├── temp/
echo    └── _internal/
echo.

echo 💡 提示:
echo    - 将整个 dist\LWF\ 文件夹打包分发即可
echo    - 用户解压后双击 LWF.exe 即可运行
echo    - 首次运行会引导下载 Whisper 模型
echo    - 在「设置」中可配置 HuggingFace 镜像加速下载
echo.

pause
