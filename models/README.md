# /models 目录

faster-whisper CTranslate2 格式模型的存放位置。

## 支持的模型

| 模型名 | 大小 | 参数量 | 推荐显存 | 适用场景 |
|--------|------|--------|----------|----------|
| `tiny` | ~75 MB | 39M | 1 GB | 极速测试 |
| `base` | ~145 MB | 74M | 1 GB | 日常使用，速度精度均衡 |
| `small` | ~484 MB | 244M | 2 GB | 较高精度 |
| `medium` | ~1.5 GB | 769M | 4 GB | 高精度 |
| `large-v3` | ~3 GB | 1550M | 6 GB | 最高精度 |

## 获取模型

### 方式 1：程序内下载（推荐）
在「模型管理」标签页中点击下载按钮，程序会自动从 HuggingFace 下载。

### 方式 2：手动放入
从 HuggingFace 下载对应的模型文件夹，整个放到这个目录下即可。
例如 `models/faster-whisper-base/` 包含 `model.bin`、`config.json` 等文件。

> 💡 国内用户如果下载慢，可在「设置」中配置 HuggingFace 镜像地址。
