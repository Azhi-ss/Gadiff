---
name: wsl-browser-open
description: Open files/URLs from WSL2 in the Windows default browser. Tested and verified approaches: cmd.exe /c start (simple) and powershell.exe Start-Process (robust, no UNC path warnings). Also covers python -m http.server for serving local files.
version: 1.0.0
triggers:
  - "wsl 打开浏览器"
  - "wsl browser"
  - "open in windows browser"
  - "从 wsl 打开网页"
---

# WSL Browser Open

Open files and URLs from WSL2 in the Windows default browser.

## Tested Approaches

### 1. cmd.exe (simplest)

```bash
cmd.exe /c start http://localhost:8888/myfile.html
```

Works but produces UNC path warning in WSL2. The warning is harmless.

### 2. PowerShell (recommended — no warnings)

```bash
powershell.exe -Command "Start-Process 'http://localhost:8888/myfile.html'"
```

Cleaner output, no UNC path issues. **Use this as the default.**

### 3. Open local .html file directly (no server needed)

```bash
# Convert WSL path to Windows path, then open
win_path=$(wslpath -w /home/dministrator/project/Gadiff/doc/fig/sgef_architecture.html)
powershell.exe -Command "Start-Process 'file:///$win_path'"
```

### 4. Serve directory then open

```bash
# Start HTTP server in background
python -m http.server 8888 -d /path/to/dir &

# Open in Windows browser
powershell.exe -Command "Start-Process 'http://localhost:8888/file.html'"
```

## WSL2 localhost mapping

WSL2 automatically forwards `localhost` from Windows to WSL2. Any server running on `localhost:PORT` in WSL2 is directly accessible from Windows browsers at the same address. No `wslhost` or extra config needed.

## Checking available tools

```bash
which cmd.exe        # /mnt/c/Windows/system32/cmd.exe — always available
which powershell.exe # /mnt/c/Windows/System32/... — always available
which wslview        # wslu package — may not be installed
```
