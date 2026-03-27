# /bin 目录

存放 FFmpeg 可执行文件和 GPU 加速所需的 CUDA DLL。

## 必需文件
- `ffmpeg.exe` — 音视频解码（用于提取音轨）
- `ffprobe.exe` — 媒体信息探测

## GPU 加速文件（可选）
使用 NVIDIA GPU 加速时需要以下 DLL（CUDA 12 + cuDNN 9）：

| 文件名 | 说明 |
|--------|------|
| `cublas64_12.dll` | cuBLAS 矩阵运算库 |
| `cublasLt64_12.dll` | cuBLAS 轻量版 |
| `cudnn_ops64_9.dll` | cuDNN 算子库 |
| `cudnn_cnn64_9.dll` | cuDNN CNN 算子 |

> 💡 不放 GPU 相关 DLL 也能正常运行，会自动回退到 CPU 模式。

## 下载来源
- **FFmpeg**: https://www.gyan.dev/ffmpeg/builds/ （essentials 版本）
- **CUDA DLL**: https://github.com/Purfview/whisper-standalone-win/releases
