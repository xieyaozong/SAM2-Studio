# GitHub Upload Notes

這個 repo 適合上傳「源碼 + 可重現打包設定」，不要把本機 runtime 和模型權重一起推上 Git。

## 建議 Commit 內容

```powershell
git add .gitignore README.md requirements.txt SAM2Studio.spec sam2_studio.py run_sam_app.bat utils scripts docs
git add sam2 setup.py pyproject.toml MANIFEST.in LICENSE LICENSE_cctorch checkpoints/download_ckpts.sh
git status --short
git diff --cached --stat
git commit -m "Add SAM2 Studio personal tool"
git push origin main
```

如果你的預設分支不是 `main`，請把最後一行換成自己的分支名。

## 不要 Commit 的內容

這些檔案已在 `.gitignore` 內：

```text
.venv/
SAM2Studio.exe
_internal/
checkpoints/*.pt
outputs/
__pycache__/
build/
dist/
```

## 分享 Windows exe

如果要讓別人直接用 exe，請把以下兩個放在同一層並壓成 zip：

```text
SAM2Studio.exe
_internal/
```

建議上傳到 GitHub Releases，不建議放進 git commit。

## Clone 後重建環境

```powershell
git clone <your-repo-url>
cd <repo>
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

然後下載 SAM2.1 checkpoints 到 `checkpoints/`。

## 重新打包

```powershell
.\scripts\build_exe.ps1
```

打包完成後，根目錄會有：

```text
SAM2Studio.exe
_internal/
```
