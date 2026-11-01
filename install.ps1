<#
.SYNOPSIS
    Installs, repairs or uninstalls MGTranscriber on Windows.

.DESCRIPTION
    - checks the NVIDIA GPU (nvidia-smi) and picks the CUDA 12.8 or the CPU build of PyTorch
    - installs uv, ffmpeg and Deno when they are missing: winget (checks hashes), otherwise pinned official
      downloads (uv, Deno: GitHub releases; ffmpeg: gyan.dev) checked against the publisher's SHA-256 file
    - creates the Python environment (uv sync)
    - asks for the HuggingFace token (speaker recognition) and optionally sets up Ollama
    - creates Desktop and Start menu shortcuts and an entry in Settings > Apps

    Running it again is safe: every step checks first and only does what is missing.
    Transcripts (output\), app data (data\), .env and settings.json are never deleted.

    Log: %LOCALAPPDATA%\mgtranscriber\logs\install-<date>.log

.EXAMPLE
    .\install.bat                      # from a downloaded copy
    .\install.ps1 -Repair              # reinstall Python packages and shortcuts
    .\install.ps1 -Uninstall           # remove environment and shortcuts, keep transcripts
    .\install.ps1 -Cpu                 # force the CPU build (no NVIDIA GPU, or for testing)

    # without a downloaded copy (read the script first, see README):
    irm https://raw.githubusercontent.com/Michal1609/MGTranscriber/main/install.ps1 | iex
    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/Michal1609/MGTranscriber/main/install.ps1))) -Cpu
#>
[CmdletBinding()]
param(
    [switch]$Repair,
    [switch]$Uninstall,
    [switch]$Cpu,                  # force the CPU build of PyTorch
    [switch]$Gpu,                  # force the CUDA build even when the check says otherwise
    [switch]$NonInteractive,       # no questions (CI, scripted installs); HF token from -HfToken or $env:HF_TOKEN
    [string]$HfToken = "",
    [ValidateSet("ask", "yes", "no")][string]$Ollama = "ask",
    [string]$OllamaModel = "gemma3:4b",
    [switch]$NoShortcuts,
    [switch]$NoAppsEntry,          # the setup .exe registers itself in Settings > Apps
    [switch]$NoToolInstall,        # only check uv/ffmpeg/Deno, never install them
    [switch]$NoWinget,             # skip winget, use the official downloads
    [switch]$NoUserPath,           # tools installed by this script go on PATH for this session only
    [string]$InstallDir = "",      # where the one-liner puts the app (default %LOCALAPPDATA%\Programs\mgtranscriber)
    [string]$ToolsDir = "",        # where fallback downloads of ffmpeg/Deno go (default %LOCALAPPDATA%\mgtranscriber\tools)
    [string]$Repo = "Michal1609/MGTranscriber"
)

# `irm ... | iex` runs this in the user's own session: remember what we change and put it back in `finally`
$SavedProgress = $ProgressPreference
$SavedErrorAction = $ErrorActionPreference
$SavedProtocol = [Net.ServicePointManager]::SecurityProtocol
Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$AppName = "mgtranscriber"                   # keep in sync with APP_NAME in mgtranscriber/__init__.py
$DisplayName = "MGTranscriber"             # shown to the user; folders, shortcuts and the uninstall entry use it
$AppData = Join-Path $env:LOCALAPPDATA $DisplayName
if (-not $ToolsDir) { $ToolsDir = Join-Path $AppData "tools" }
$LogDir = Join-Path $AppData "logs"
$LogFile = Join-Path $LogDir ("install-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss"))
$UninstallKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$AppName"
$PyannoteModel = "pyannote/speaker-diarization-community-1"
$MinDriverCuda12 = [version]"528.33"       # oldest Windows driver that runs CUDA 12.x
$RecommendedDriver = [version]"570.65"     # CUDA 12.8 GA; required for RTX 50xx
# Fallback downloads (used only when winget is missing or fails): pinned versions, hash-checked
$UvVersion = "0.12.24"                     # same as UV_VERSION in the Dockerfile
$DenoVersion = "2.9.7"                     # same as DENO_VERSION in the Dockerfile
$FfmpegVersion = "9.0.2"                   # gyan.dev essentials build
$MinComputeCap = [version]"6.1"            # oldest GPU the cu128 PyTorch wheels are built for (Pascal)

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# --- output ------------------------------------------------------------------------------

function Write-Log([string]$Message) {
    Add-Content -Path $LogFile -Value ("{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $Message) -Encoding UTF8
}
function Say([string]$Message) { Write-Host $Message; Write-Log $Message }
function Step([string]$Message) { Write-Host ""; Write-Host "==> $Message" -ForegroundColor Cyan; Write-Log "==> $Message" }
function Ok([string]$Message) { Write-Host "    OK  $Message" -ForegroundColor Green; Write-Log "OK  $Message" }
function Warn([string]$Message) { Write-Host "    !   $Message" -ForegroundColor Yellow; Write-Log "WARN $Message" }

class InstallError : System.Exception {
    [string]$Hint
    InstallError([string]$Message, [string]$Hint) : base($Message) { $this.Hint = $Hint }
}
function Fail([string]$Message, [string]$Hint = "") { throw [InstallError]::new($Message, $Hint) }

function Ask([string]$Question, [string]$Default = "") {
    if ($NonInteractive) { return $Default }
    $suffix = if ($Default) { " [$Default]" } else { "" }
    $answer = Read-Host "    $Question$suffix"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer.Trim()
}
# Like Ask, but the typed text is not echoed (tokens).
function AskSecret([string]$Question) {
    if ($NonInteractive) { return "" }
    $secure = Read-Host "    $Question" -AsSecureString
    $plain = (New-Object System.Net.NetworkCredential("", $secure)).Password
    return $plain.Trim()
}
function AskYes([string]$Question, [bool]$Default) {
    $answer = Ask "$Question (y/n)" $(if ($Default) { "y" } else { "n" })
    return $answer -match '^(y|yes|a|ano)$'
}

# Runs a program, sends its output to the log (and the console), fails with a readable message.
function Invoke-Logged([string]$File, [string[]]$Arguments, [string]$What, [string]$Hint = "") {
    Write-Log "> $File $($Arguments -join ' ')"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"    # native programs write progress to stderr
    try {
        & $File @Arguments 2>&1 | ForEach-Object { $line = "$_"; Write-Host "      $line"; Write-Log "      $line" }
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previous }
    if ($code -ne 0) { Fail "$What failed (exit code $code)." $Hint }
}

# --- environment helpers -----------------------------------------------------------------

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $extra = @($env:Path -split ";" | Where-Object { $_ -and ("$machine;$user" -split ";") -notcontains $_ })
    $env:Path = (@($machine, $user) + $extra | Where-Object { $_ }) -join ";"
}

function Add-ToPath([string]$Dir) {
    if (($env:Path -split ";") -notcontains $Dir) { $env:Path = "$Dir;$env:Path" }
    if ($NoUserPath) { return }
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    if ((("$user" -split ";") -notcontains $Dir)) {
        [Environment]::SetEnvironmentVariable("Path", (@($Dir) + @("$user" -split ";" | Where-Object { $_ })) -join ";", "User")
        Say "    added $Dir to your PATH"
    }
}

function Test-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

function Test-Winget { return (-not $NoWinget) -and (Test-Command "winget") }

function Install-WithWinget([string]$Id) {
    if (-not (Test-Winget)) { return $false }
    Say "    winget install $Id"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements --scope user 2>&1 |
            ForEach-Object { Write-Log "      $_" }
        $code = $LASTEXITCODE
        if ($code -ne 0) {  # some packages have no per-user installer
            & winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements 2>&1 |
                ForEach-Object { Write-Log "      $_" }
            $code = $LASTEXITCODE
        }
    } finally { $ErrorActionPreference = $previous }
    Update-SessionPath
    # -1978335189: already installed / no newer version
    if ($code -eq 0 -or $code -eq -1978335189) { return $true }
    Warn "winget could not install $Id (exit code $code), trying the official download"
    return $false
}

function Get-HttpStatus($ErrorRecord) {
    $response = $ErrorRecord.Exception.PSObject.Properties["Response"]
    if ($response -and $response.Value) { return [int]$response.Value.StatusCode }
    return $null
}

function Save-Download([string]$Url, [string]$Target) {
    Write-Log "download $Url -> $Target"
    try { Invoke-WebRequest -Uri $Url -OutFile $Target -UseBasicParsing }
    catch { Fail "Download failed: $Url ($($_.Exception.Message))" "Check the internet connection or a proxy/firewall, then run the installer again." }
}

# Downloads a file and checks its SHA-256 against the hash the publisher posts next to it
# (the file may hold only the hash, "hash  name", or a PowerShell Get-FileHash dump).
function Save-Verified([string]$Url, [string]$HashUrl, [string]$Target) {
    Save-Download $Url $Target
    try { $text = (New-Object Net.WebClient).DownloadString($HashUrl) }
    catch { Fail "Could not download the checksum $HashUrl ($($_.Exception.Message))." "Check the internet connection, or install the tool with winget." }
    if ($text -notmatch '(?i)\b([0-9a-f]{64})\b') { Fail "No SHA-256 found in $HashUrl." "Install the tool with winget instead." }
    $expected = $Matches[1].ToLowerInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 -Path $Target).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        Remove-Item $Target -Force -ErrorAction SilentlyContinue
        Fail "The download of $Url does not match its published SHA-256 (expected $expected, got $actual)." "Try again; if it keeps failing, install the tool with winget."
    }
    Write-Log "sha256 ok $actual $Url"
}

# --- project location --------------------------------------------------------------------

function Test-ProjectRoot([string]$Dir) {
    return $Dir -and (Test-Path (Join-Path $Dir "pyproject.toml")) -and (Test-Path (Join-Path $Dir "$AppName\__init__.py"))
}

function Get-ProjectRoot {
    if (Test-ProjectRoot $PSScriptRoot) { return $PSScriptRoot }
    if (Test-ProjectRoot (Get-Location).Path) { return (Get-Location).Path }
    $target = if ($InstallDir) { $InstallDir } else { Join-Path $env:LOCALAPPDATA "Programs\$AppName" }
    if ($Uninstall -or $Repair) {
        if (Test-ProjectRoot $target) { return $target }
        Fail "No installation found in $target." "Run the installer from the project folder, or pass -InstallDir."
    }
    Step "Downloading $DisplayName into $target"
    $zipUrl = "https://github.com/$Repo/archive/refs/heads/main.zip"
    try {
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
        $zipUrl = "https://github.com/$Repo/archive/refs/tags/$($release.tag_name).zip"
        Say "    latest release $($release.tag_name)"
    } catch { Warn "no release found, using the main branch" }
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "$AppName-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        Save-Download $zipUrl (Join-Path $tmp "src.zip")
        Expand-Archive -Path (Join-Path $tmp "src.zip") -DestinationPath (Join-Path $tmp "x") -Force
        $src = Get-ChildItem (Join-Path $tmp "x") -Directory | Select-Object -First 1
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        # an upgrade replaces the code folders, so modules deleted upstream do not linger; output\, data\, .env
        # and settings.json are not in the archive and are never touched
        foreach ($dir in Get-ChildItem $src.FullName -Directory) {
            $old = Join-Path $target $dir.Name
            if (($dir.Name -notin @("output", "data", ".venv")) -and (Test-Path $old)) { Remove-Item $old -Recurse -Force }
        }
        Copy-Item -Path (Join-Path $src.FullName "*") -Destination $target -Recurse -Force
    } finally { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
    Ok "app files in $target"
    return $target
}

# --- tools -------------------------------------------------------------------------------

function Install-Uv {
    if (Test-Command "uv") { Ok "uv $((& uv --version) -replace '^uv ', '')"; return }
    if ($NoToolInstall) { Fail "uv is missing." "Install it: winget install astral-sh.uv  (https://docs.astral.sh/uv/)" }
    if (-not (Install-WithWinget "astral-sh.uv") -or -not (Test-Command "uv")) {
        Say "    downloading uv $UvVersion from github.com/astral-sh/uv (SHA-256 checked)"
        $base = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
        $bin = if ($env:UV_INSTALL_DIR) { $env:UV_INSTALL_DIR } else { Join-Path $env:USERPROFILE ".local\bin" }
        $tmp = Join-Path ([IO.Path]::GetTempPath()) "uv-$([guid]::NewGuid().ToString('N'))"
        New-Item -ItemType Directory -Path $tmp | Out-Null
        try {
            Save-Verified $base "$base.sha256" (Join-Path $tmp "uv.zip")
            Expand-Archive -Path (Join-Path $tmp "uv.zip") -DestinationPath (Join-Path $tmp "x") -Force
            New-Item -ItemType Directory -Force -Path $bin | Out-Null
            Get-ChildItem (Join-Path $tmp "x") -Recurse -Filter "*.exe" | Copy-Item -Destination $bin -Force
        } finally { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
        Update-SessionPath
        Add-ToPath $bin
    }
    if (-not (Test-Command "uv")) { Fail "uv was installed but is not on PATH." "Open a new terminal and run the installer again." }
    Ok "uv $((& uv --version) -replace '^uv ', '')"
}

function Install-Ffmpeg {
    if (Test-Command "ffmpeg") { Ok "ffmpeg found"; return }
    $local = Join-Path $ToolsDir "ffmpeg\bin"
    if (Test-Path (Join-Path $local "ffmpeg.exe")) { Add-ToPath $local; Ok "ffmpeg in $local"; return }
    if ($NoToolInstall) { Fail "ffmpeg is missing." "Install it: winget install Gyan.FFmpeg" }
    if ((Install-WithWinget "Gyan.FFmpeg") -and (Test-Command "ffmpeg")) { Ok "ffmpeg installed"; return }
    Say "    downloading the ffmpeg $FfmpegVersion essentials build from gyan.dev (SHA-256 checked)"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "ffmpeg-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $url = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-$FfmpegVersion-essentials_build.zip"
        Save-Verified $url "$url.sha256" (Join-Path $tmp "ffmpeg.zip")
        Expand-Archive -Path (Join-Path $tmp "ffmpeg.zip") -DestinationPath $tmp -Force
        $bin = Get-ChildItem $tmp -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
        if (-not $bin) { Fail "The ffmpeg download did not contain ffmpeg.exe." "Install ffmpeg manually: winget install Gyan.FFmpeg" }
        New-Item -ItemType Directory -Force -Path $local | Out-Null
        Copy-Item -Path (Join-Path $bin.DirectoryName "*") -Destination $local -Force
    } finally { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
    Add-ToPath $local
    Ok "ffmpeg in $local"
}

function Get-DenoVersion {
    if (-not (Test-Command "deno")) { return $null }
    $line = (& deno --version 2>$null | Select-Object -First 1)
    if ($line -match 'deno (\d+\.\d+\.\d+)') { return [version]$Matches[1] }
    return $null
}

function Install-Deno {
    # yt-dlp needs a JavaScript runtime for YouTube (Deno 2.3+, or Node.js 22+)
    $deno = Get-DenoVersion
    if ($deno -and $deno -ge [version]"2.3.0") { Ok "Deno $deno"; return }
    if (Test-Command "node") {
        $node = (& node --version) -replace '^v', ''
        if ([version]($node -replace '-.*$', '') -ge [version]"22.0.0") { Ok "Node.js $node (used by yt-dlp instead of Deno)"; return }
    }
    $local = Join-Path $ToolsDir "deno"
    if (Test-Path (Join-Path $local "deno.exe")) { Add-ToPath $local; Ok "Deno in $local"; return }
    if ($NoToolInstall) { Warn "Deno is missing: downloading from YouTube will not work (winget install DenoLand.Deno)"; return }
    if ((Install-WithWinget "DenoLand.Deno") -and (Get-DenoVersion)) { Ok "Deno $(Get-DenoVersion)"; return }
    Say "    downloading Deno $DenoVersion from github.com/denoland/deno (SHA-256 checked)"
    $tmp = Join-Path ([IO.Path]::GetTempPath()) "deno-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        $url = "https://github.com/denoland/deno/releases/download/v$DenoVersion/deno-x86_64-pc-windows-msvc.zip"
        Save-Verified $url "$url.sha256sum" (Join-Path $tmp "deno.zip")
        New-Item -ItemType Directory -Force -Path $local | Out-Null
        Expand-Archive -Path (Join-Path $tmp "deno.zip") -DestinationPath $local -Force
    } finally { Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue }
    Add-ToPath $local
    Ok "Deno $(Get-DenoVersion) in $local"
}

# --- GPU and PyTorch build ---------------------------------------------------------------

function Get-Gpu {
    $smi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if (-not $smi) {
        $fallback = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
        if (Test-Path $fallback) { $smi = $fallback } else { return $null }
    }
    try {
        $rows = & $smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader,nounits 2>$null
    } catch { return $null }
    if ($LASTEXITCODE -ne 0 -or -not $rows) {
        # drivers older than ~510 do not know the compute_cap field; they are too old for CUDA 12 anyway
        return [pscustomobject]@{ Name = "NVIDIA GPU (details unknown)"; Driver = [version]"0.0"; ComputeCap = [version]"0.0"; MemoryMiB = 0 }
    }
    $best = $null
    foreach ($row in @($rows)) {
        $f = "$row" -split ",\s*"
        if ($f.Count -lt 4) { continue }
        try {
            $gpu = [pscustomobject]@{ Name = $f[0].Trim(); Driver = [version]($f[1].Trim()); ComputeCap = [version]($f[2].Trim()); MemoryMiB = [int]$f[3].Trim() }
        } catch { Write-Log "nvidia-smi row not understood: $row"; continue }  # e.g. "[N/A]" fields
        if (-not $best -or $gpu.MemoryMiB -gt $best.MemoryMiB) { $best = $gpu }
    }
    if (-not $best) { return [pscustomobject]@{ Name = "NVIDIA GPU (details unknown)"; Driver = [version]"0.0"; ComputeCap = [version]"0.0"; MemoryMiB = 0 } }
    return $best
}

function Select-TorchBuild {
    if ($Cpu -and $Gpu) { Fail "Use either -Cpu or -Gpu, not both." }
    if ($Cpu) { Say "    CPU build forced with -Cpu"; return "cpu" }
    $gpu = Get-Gpu
    if (-not $gpu) {
        if ($Gpu) { Warn "-Gpu given but nvidia-smi found no GPU; installing the CUDA build anyway"; return "cu128" }
        Warn "no NVIDIA GPU found (nvidia-smi). Installing the CPU build: transcription works, but much slower."
        Say  "        Have an NVIDIA card? Install the driver from https://www.nvidia.com/drivers and run the installer again."
        return "cpu"
    }
    Say ("    {0}, driver {1}, compute capability {2}, {3:N1} GB VRAM" -f $gpu.Name, $gpu.Driver, $gpu.ComputeCap, ($gpu.MemoryMiB / 1024))
    if ($Gpu) { return "cu128" }
    if ($gpu.Driver -eq [version]"0.0") {
        Warn "nvidia-smi could not report the GPU; the driver is probably too old. Using the CPU build for now."
        Say  "        Update the driver (https://www.nvidia.com/drivers), then run: install.bat -Repair"
        return "cpu"
    }
    if ($gpu.ComputeCap -lt $MinComputeCap) {
        Warn "this GPU is too old for current PyTorch (needs compute capability $MinComputeCap+); using the CPU build"
        return "cpu"
    }
    if ($gpu.Driver -lt $MinDriverCuda12) {
        Warn "driver $($gpu.Driver) is too old for CUDA 12 (needs $MinDriverCuda12+). Using the CPU build for now."
        Say  "        Update the driver (https://www.nvidia.com/drivers), then run: install.bat -Repair"
        return "cpu"
    }
    if ($gpu.Driver -lt $RecommendedDriver) { Warn "driver $($gpu.Driver) is older than $RecommendedDriver; if the GPU is not used, update the driver" }
    if ($gpu.MemoryMiB -lt 5500) { Warn "less than 6 GB VRAM: use a smaller model (large-v3-turbo) or batch size 4 if you run out of memory" }
    return "cu128"
}

function Get-UvGroupArgs([string]$Build) {
    if ($Build -eq "cpu") { return @("--no-group", "cu128", "--group", "cpu") }
    return @()
}

# --- configuration -----------------------------------------------------------------------

function Get-EnvValue([string]$File, [string]$Key) {
    if (-not (Test-Path $File)) { return "" }
    foreach ($line in Get-Content $File -Encoding UTF8) {
        if ($line -match "^\s*$Key\s*=\s*(.*)$") { return $Matches[1].Trim().Trim('"').Trim("'") }
    }
    return ""
}

function Set-EnvValue([string]$File, [string]$Key, [string]$Value) {
    $lines = @()
    if (Test-Path $File) { $lines = @(Get-Content $File -Encoding UTF8) }
    $found = $false
    $lines = @($lines | ForEach-Object { if ($_ -match "^\s*$Key\s*=") { $found = $true; "$Key=$Value" } else { $_ } })
    if (-not $found) { $lines += "$Key=$Value" }
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllLines($File, [string[]]$lines, $utf8)
}

# Returns "ok", "invalid", "not-accepted" or "unknown" (no network).
function Test-HfToken([string]$Token) {
    $headers = @{ Authorization = "Bearer $Token" }
    try { Invoke-RestMethod -Uri "https://huggingface.co/api/whoami-v2" -Headers $headers -UseBasicParsing | Out-Null }
    catch {
        if ((Get-HttpStatus $_) -eq 401) { return "invalid" }
        return "unknown"
    }
    try {
        Invoke-WebRequest -Method Head -Uri "https://huggingface.co/$PyannoteModel/resolve/main/config.yaml" -Headers $headers -UseBasicParsing | Out-Null
        return "ok"
    } catch {
        $status = Get-HttpStatus $_
        if ($status -eq 401 -or $status -eq 403) { return "not-accepted" }
        return "unknown"
    }
}

function Set-HfToken([string]$Root) {
    $envFile = Join-Path $Root ".env"
    if (-not (Test-Path $envFile)) {
        $example = Join-Path $Root ".env.example"
        if (Test-Path $example) { Copy-Item $example $envFile } else { New-Item -ItemType File -Path $envFile | Out-Null }
        Say "    created .env"
    }
    $token = if ($HfToken) { $HfToken } elseif ($env:HF_TOKEN) { $env:HF_TOKEN } else { Get-EnvValue $envFile "HF_TOKEN" }
    while ($true) {
        if (-not $token) {
            if ($NonInteractive) { Warn "no HF_TOKEN: speaker recognition is off until you add it to .env"; return }
            Say ""
            Say "    Speaker recognition (who speaks when) uses pyannote models from HuggingFace. One-time setup:"
            Say "      1. Create a free account and log in:   https://huggingface.co/join"
            Say "      2. Accept the model conditions:        https://huggingface.co/$PyannoteModel"
            Say "      3. Create a token of type 'Read':      https://huggingface.co/settings/tokens"
            $token = AskSecret "Paste the token (hf_..., typing is hidden), or press Enter to skip"
            if (-not $token) { Warn "skipped: transcription works, speakers only after you add HF_TOKEN to .env"; return }
        }
        $state = Test-HfToken $token
        switch ($state) {
            "ok"           { Set-EnvValue $envFile "HF_TOKEN" $token; Ok "HuggingFace token works and the pyannote conditions are accepted"; return }
            "unknown"      { Set-EnvValue $envFile "HF_TOKEN" $token; Warn "could not check the token (offline?); saved it to .env anyway"; return }
            "not-accepted" {
                Set-EnvValue $envFile "HF_TOKEN" $token
                Warn "token saved, but the pyannote conditions are not accepted yet: open https://huggingface.co/$PyannoteModel and click 'Agree'"
                if ($NonInteractive -or -not (AskYes "Check again after you accepted them?" $true)) { return }
            }
            "invalid"      {
                Warn "HuggingFace rejected this token"
                if ($NonInteractive) { return }
                $token = ""
            }
        }
    }
}

function Set-Ollama {
    $choice = $Ollama
    if ($choice -eq "ask") {
        if ($NonInteractive) { $choice = "no" }
        else {
            Say ""
            Say "    Optional: summaries, chapters and questions over transcripts with a local LLM (Ollama, ~4 GB download)."
            $choice = if (AskYes "Set up Ollama with $OllamaModel?" (Test-Command "ollama")) { "yes" } else { "no" }
        }
    }
    if ($choice -ne "yes") { Say "    Ollama skipped (you can run the installer again or see docs\LLM.md)"; return }
    if (-not (Test-Command "ollama")) {
        if ($NoToolInstall -or -not (Install-WithWinget "Ollama.Ollama") -or -not (Test-Command "ollama")) {
            $exe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
            if (Test-Path $exe) { Add-ToPath (Split-Path $exe) }
            else { Warn "could not install Ollama; get it from https://ollama.com/download and run: ollama pull $OllamaModel"; return }
        }
    }
    foreach ($model in @($OllamaModel, "bge-m3")) {
        $listed = (& ollama list 2>$null) -join "`n"
        if ($listed -match [regex]::Escape($model)) { Ok "Ollama model $model"; continue }
        try { Invoke-Logged "ollama" @("pull", $model) "ollama pull $model" "Start Ollama (it runs in the tray) and run: ollama pull $model" }
        catch { Warn $_.Exception.Message; return }
        Ok "Ollama model $model"
    }
}

# --- shortcuts and the Apps entry --------------------------------------------------------

function Get-ShortcutPaths {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $programs = [Environment]::GetFolderPath("Programs")
    return @((Join-Path $desktop "$DisplayName.lnk"), (Join-Path $programs "$DisplayName.lnk"))
}

function New-Shortcuts([string]$Root) {
    $shell = New-Object -ComObject WScript.Shell
    $icon = Join-Path $Root "static\favicon.ico"
    foreach ($path in Get-ShortcutPaths) {
        $link = $shell.CreateShortcut($path)
        $link.TargetPath = Join-Path $Root "web.bat"
        $link.WorkingDirectory = $Root
        $link.Description = "$DisplayName - local transcription (web UI)"
        $link.WindowStyle = 7  # minimized console; the browser opens by itself
        if (Test-Path $icon) { $link.IconLocation = $icon } else { $link.IconLocation = "$env:SystemRoot\System32\imageres.dll,103" }
        $link.Save()
    }
    Ok "shortcuts on the Desktop and in the Start menu"
}

function Register-App([string]$Root, [string]$Build) {
    $version = "0.0.0"
    $init = Get-Content (Join-Path $Root "$AppName\__init__.py") -Raw
    if ($init -match '__version__\s*=\s*"([^"]+)"') { $version = $Matches[1] }
    $ps = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
    $script = Join-Path $Root "install.ps1"
    New-Item -Path $UninstallKey -Force | Out-Null
    $values = @{
        DisplayName = $DisplayName; DisplayVersion = $version; Publisher = "Michal Grznar"
        InstallLocation = $Root; NoModify = 1; NoRepair = 1
        UninstallString = "`"$ps`" -NoProfile -ExecutionPolicy Bypass -File `"$script`" -Uninstall"
        Comments = "PyTorch build: $Build"
    }
    foreach ($k in $values.Keys) { New-ItemProperty -Path $UninstallKey -Name $k -Value $values[$k] -Force | Out-Null }
}

# --- actions -----------------------------------------------------------------------------

function Invoke-Install([string]$Root) {
    Step "Tools"
    Update-SessionPath
    Install-Uv
    Install-Ffmpeg
    Install-Deno

    Step "GPU"
    $build = Select-TorchBuild
    $marker = Join-Path $Root ".torch-cpu"   # web.bat/start.bat read it so `uv run` keeps the CPU build
    if ($build -eq "cpu") { Set-Content -Path $marker -Value "PyTorch CPU build, written by install.ps1" -Encoding ASCII }
    elseif (Test-Path $marker) { Remove-Item $marker }
    Ok "PyTorch build: $build"

    Step "Python environment (uv sync; the first run downloads 3-5 GB)"
    Push-Location $Root
    try {
        $syncArgs = @("sync", "--frozen", "--no-dev") + (Get-UvGroupArgs $build)
        if ($Repair) { $syncArgs += "--reinstall" }
        Invoke-Logged "uv" $syncArgs "uv sync" "See the log for the reason. A network error: run the installer again, it continues where it stopped."
    } finally { Pop-Location }
    Ok "environment in $(Join-Path $Root '.venv')"

    Step "HuggingFace token (speaker recognition)"
    Set-HfToken $Root

    Step "Local LLM (optional)"
    Set-Ollama

    if (-not $NoShortcuts) {
        Step "Shortcuts"
        New-Shortcuts $Root
    }
    if (-not $NoAppsEntry) { Register-App $Root $build }

    Step "Check"
    Push-Location $Root
    try {
        $python = Join-Path $Root ".venv\Scripts\python.exe"
        if (Test-Path (Join-Path $Root "$AppName\diagnostics.py")) {
            Invoke-Logged $python @("-m", "$AppName.diagnostics") "The self-check" "See the messages above."
        } else {
            $check = "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
            Invoke-Logged $python @("-c", $check) "The PyTorch check" "Run: install.bat -Repair"
        }
    } finally { Pop-Location }

    Write-Host ""
    Write-Host "Done. Start the web UI with the '$DisplayName' shortcut or web.bat (http://127.0.0.1:8000)." -ForegroundColor Green
    Write-Log "done"
}

function Invoke-Uninstall([string]$Root) {
    Step "Uninstalling $DisplayName from $Root"
    foreach ($path in Get-ShortcutPaths) { if (Test-Path $path) { Remove-Item $path -Force; Say "    removed $path" } }
    if (Test-Path $UninstallKey) { Remove-Item $UninstallKey -Recurse -Force }
    foreach ($name in @(".venv", ".torch-cpu")) {
        $path = Join-Path $Root $name
        if (Test-Path $path) {
            try { Remove-Item $path -Recurse -Force; Say "    removed $path" }
            catch { Fail "Could not delete $path ($($_.Exception.Message))." "Close the web UI and any terminal using it, then run the uninstall again." }
        }
    }
    Ok "removed the Python environment and shortcuts"
    Say ""
    Say "    Kept (delete by hand if you do not need them):"
    foreach ($name in @("output", "data", ".env", "settings.json")) {
        $path = Join-Path $Root $name
        if (Test-Path $path) { Say "      $path" }
    }
    Say "      $AppData  (logs, downloaded tools)"
    Say "      downloaded models: $env:USERPROFILE\.cache\huggingface"
    Say "    uv, ffmpeg, Deno and Ollama stay installed (winget uninstall <name> removes them)."
}

# --- main --------------------------------------------------------------------------------

try {
    $arguments = ($PSBoundParameters.GetEnumerator() | ForEach-Object { "-$($_.Key) $($_.Value)" }) -join " "
    Write-Log ("install.ps1 " + ($arguments -replace '(-HfToken )\S+', '$1***'))

    Write-Log "PowerShell $($PSVersionTable.PSVersion), $([Environment]::OSVersion.VersionString)"
    if ($Repair -and $Uninstall) { Fail "Use either -Repair or -Uninstall." }
    if (-not [Environment]::Is64BitOperatingSystem) { Fail "$DisplayName needs 64-bit Windows." }
    $root = Get-ProjectRoot
    Write-Log "root $root"
    if ($Uninstall) { Invoke-Uninstall $root } else { Invoke-Install $root }
    Write-Host "Log: $LogFile" -ForegroundColor DarkGray
    $exitCode = 0
} catch {
    $err = $_.Exception
    Write-Host ""
    Write-Host "ERROR: $($err.Message)" -ForegroundColor Red
    if ($err -is [InstallError] -and $err.Hint) { Write-Host "       $($err.Hint)" -ForegroundColor Yellow }
    Write-Host "       Details are in the log: $LogFile" -ForegroundColor DarkGray
    Write-Log "ERROR $($err.Message)"
    Write-Log ($_ | Out-String)
    $exitCode = 1
} finally {
    Set-StrictMode -Off
    $ErrorActionPreference = $SavedErrorAction
    $ProgressPreference = $SavedProgress
    [Net.ServicePointManager]::SecurityProtocol = $SavedProtocol
}
# `irm ... | iex` runs this in the user's own session: `exit` would close their window
if ($PSCommandPath) { exit $exitCode }
