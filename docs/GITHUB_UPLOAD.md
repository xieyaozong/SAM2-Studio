# GitHub Upload Notes

This repository is now structured as a personal application repo rather than a fork-style copy of Meta's SAM2 project. The upstream `sam2` package is installed through `requirements.txt`.

## Commit Source Code

```powershell
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare SAM2 Studio personal tool"
git push origin main
```

Use your current branch name instead of `main` if needed.

## Do Not Commit Local Artifacts

These are ignored by `.gitignore`:

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

## Sharing the Windows App

To distribute a ready-to-run Windows build, zip these together:

```text
SAM2Studio.exe
_internal/
```

Upload the zip to GitHub Releases or another file host. Avoid committing it directly; the runtime is several gigabytes because it includes PyTorch, PySide6, SAM2, and optionally checkpoint files.

## Rebuild After Clone

```powershell
git clone <your-repo-url>
cd <repo>
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
.\scripts\download_checkpoints.ps1
.\scripts\build_exe.ps1
```
