# SAM2 Studio

SAM2 Studio 是一個我為影像分割標註流程整理出來的小工具，基於 Meta 的 [Segment Anything Model 2](https://github.com/facebookresearch/sam2) 做互動式影像 segmentation，並加入針對目前測試圖集的 Hough / foreground 前處理、YOLO polygon 匯出、批次輸出和 Windows exe 打包流程。

這個 repo 主要目標是方便自己在 Windows 上開 GUI 標註圖片，不是原始 SAM2 專案的替代品。

## 功能

- PySide6 GUI，可用滑鼠點選前景 / 背景點互動生成 mask。
- 支援 single image / folder workflow。
- Hough / foreground preprocessing：
  - `Preview Hough` 只預覽結果。
  - `Use Hough For SAM` 可直接執行前處理並套用到 SAM，不必先 preview。
  - 支援 full masked image 或 center crop。
- 匯出 YOLO segmentation label、mask label、overlay preview、object metadata。
- 可打包成無 console 視窗的 Windows exe。

## 專案結構

```text
sam2_studio.py          # GUI / CLI 入口
run_sam_app.bat         # 開發時快速啟動，會使用 .venv
SAM2Studio.spec         # PyInstaller 打包設定
requirements.txt        # 工具所需 Python 依賴
utils/                  # SAM2 Studio GUI、IO、export、preprocess 程式碼
sam2/                   # Meta SAM2 原始模型程式碼
checkpoints/            # 本機模型權重位置，*.pt 不上傳 git
scripts/build_exe.ps1   # Windows 打包腳本
docs/GITHUB_UPLOAD.md   # 上傳 GitHub 和 release 建議
```

本機打包後會出現：

```text
SAM2Studio.exe
_internal/
```

這兩個是本機 runtime，不會上傳 git。如果要分享 exe，請把它們一起壓成 zip 放到 GitHub Releases 或其他檔案空間。

## 安裝

建議使用 Python 3.11。

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果需要指定 CUDA 版 PyTorch，請先依照 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/) 安裝對應版本，再執行 `pip install -r requirements.txt`。

## 下載 SAM2 Checkpoints

模型權重很大，不要 commit 到 git。下載到 `checkpoints/` 後再執行工具。

Windows 可用 Git Bash 執行：

```bash
cd checkpoints
./download_ckpts.sh
```

或自行下載 SAM2.1 權重並放成：

```text
checkpoints/sam2.1_hiera_tiny.pt
checkpoints/sam2.1_hiera_small.pt
checkpoints/sam2.1_hiera_base_plus.pt
checkpoints/sam2.1_hiera_large.pt
```

## 執行 GUI

開發模式：

```powershell
.\.venv\Scripts\python.exe sam2_studio.py --gui
```

或直接雙擊：

```text
run_sam_app.bat
```

如果已經打包，可直接雙擊根目錄的：

```text
SAM2Studio.exe
```

注意：`SAM2Studio.exe` 必須和 `_internal/` 放在同一層。

## CLI Dry Run

```powershell
.\.venv\Scripts\python.exe sam2_studio.py `
  --input C:\path\to\images `
  --output C:\path\to\output `
  --recursive `
  --dry-run
```

## 打包 Windows exe

```powershell
.\scripts\build_exe.ps1
```

打包腳本會使用 `SAM2Studio.spec` 建置，並把結果整理到 repo 根目錄：

```text
SAM2Studio.exe
_internal/
```

## GitHub 上傳原則

請 commit 源碼與設定，不要 commit 本機產物：

- 不上傳：`.venv/`、`SAM2Studio.exe`、`_internal/`、`checkpoints/*.pt`、`outputs/`
- 可上傳：`sam2_studio.py`、`utils/`、`sam2/`、`requirements.txt`、`SAM2Studio.spec`、`scripts/`、`docs/`

更多指令可看 [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md)。

## Attribution

This project includes and builds on Meta's Segment Anything Model 2 source code. SAM2 code and checkpoints are subject to their original licenses and terms. See [LICENSE](LICENSE), [LICENSE_cctorch](LICENSE_cctorch), and the upstream [facebookresearch/sam2](https://github.com/facebookresearch/sam2) repository.
