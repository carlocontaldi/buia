# Installs or updates BUIA, whichever applies: no core\.env yet means install, otherwise update (git pull, rebuild, restart).
# Run .\setup.cmd in the folder above (it calls this script with the execution policy bypassed), or .\setup.ps1 from this folder.
# Add -Tailscale to also expose the UI to your tailnet (phone access).
# Add -Sd to also set up image/video generation (clones stable-diffusion.cpp and builds the sd service; the first build is long).
# Add -CogVideo to also build the cogvideo service (CogVideoX-2b text-to-video, PyTorch; the first build downloads several GB).
# Add -Llama to also look at the newest llama.cpp server image. docker-compose.yml pins the image by digest (a known working build),
# so this only shows what the moving tag points to now. To adopt it, paste its digest into docker-compose.yml (or LLAMA_IMAGE in .env) and run this script again.
# Add -NoPull to update without git pull (you have local commits, or this is not a git clone).
# An existing .env or BUIA.md is never overwritten. Linux has setup.sh, which does the same.
param([switch]$Tailscale, [switch]$Sd, [switch]$CogVideo, [switch]$Llama, [switch]$NoPull)

$SdTag = "master-920-2f88688"   # stable-diffusion.cpp release the sd service is built from (keep equal to SD_TAG in setup.sh)
$LlamaDefault = "ghcr.io/ggml-org/llama.cpp:server-cuda13"   # keep equal to the tag in docker-compose.yml (and to LLAMA_DEFAULT in setup.sh)

Set-Location $PSScriptRoot

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function New-Password {   # 32 hex characters from the OS random number generator (hex: no $ or # for Compose to trip over)
    $b = New-Object byte[] 16
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
    -join ($b | ForEach-Object { $_.ToString('x2') })
}

function Get-EnvValue($name) { (@(Get-Content .env -Encoding UTF8) | Where-Object { $_ -match "^$name=\S" } | Select-Object -First 1) -replace "^$name=", '' }

function Set-EnvValue($name, $value) {
    # Only that one line changes (or is added). .env is read and written as UTF-8, which is what Compose reads: Set-Content -Encoding ascii turned every
    # accented character (in a path, say) into "?", and Windows PowerShell 5.1 reads a file without a BOM as ANSI.
    $utf8 = New-Object Text.UTF8Encoding($false)
    $envPath = Join-Path $PSScriptRoot '.env'
    $text = [IO.File]::ReadAllText($envPath, $utf8)
    if ($text -match "(?m)^$name=") { $new = $text -replace "(?m)^$name=[^\r\n]*", "$name=$value" } else { $new = $text.TrimEnd() + "`n$name=$value`n" }
    if ($new -ne $text) { [IO.File]::WriteAllText($envPath, $new, $utf8) }
}

# 1. Install or update?
$mode = if (Test-Path .env) { 'update' } else { 'install' }
Write-Host "BUIA $mode ($((Resolve-Path '..').Path))"

# 2. Update: get the newest version first, so the rest of this run is about it. A zip download has no .git and cannot update.
if ($mode -eq 'update' -and -not $NoPull) {
    git -C .. rev-parse --is-inside-work-tree *> $null
    if ($LASTEXITCODE -eq 0) {
        git pull --ff-only
        if ($LASTEXITCODE -ne 0) { throw "git pull failed (exit code $LASTEXITCODE). Fix that, or run .\setup.cmd -NoPull to rebuild what you have." }
    } else {
        Write-Host "Not a git clone: nothing to pull."
    }
}

# 3. Docker must be running
docker info *> $null
Check "Docker (is Docker Desktop running?)"

# 4. Warn early if containers cannot see the GPU
docker run --rm --gpus all ubuntu nvidia-smi *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Docker cannot see the NVIDIA GPU. Check the NVIDIA driver and Docker Desktop's WSL2 backend before continuing."
}

# 5. .env: created once from .env.example, never overwritten
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "Created .env"
}

# 5a. The pinned llama.cpp image is built for CUDA 13, which needs NVIDIA driver 580 or newer
$driver = ''
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $driver = [string](nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1) }
$driverMajor = 0
if ([int]::TryParse(($driver -split '\.')[0], [ref]$driverMajor) -and $driverMajor -lt 580 -and -not $env:LLAMA_IMAGE -and -not (Get-EnvValue 'LLAMA_IMAGE')) {
    Write-Warning "NVIDIA driver $driver is older than 580, which the pinned CUDA 13 llama.cpp image needs. Put LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda in core\.env (README > Requirements), then run this again."
}

# 5b. Web UI password: a random one if .env has none, never overwritten
$envLines = @(Get-Content .env)
if (-not ($envLines -match '^BUIA_PASSWORD=')) {
    $pw = New-Password
    Add-Content .env "`nBUIA_PASSWORD=$pw" -Encoding ascii
    Write-Host "Web UI password: $pw   (saved in .env as BUIA_PASSWORD; change it there)"
} elseif (-not ($envLines -match '^BUIA_PASSWORD=.+')) {
    Write-Warning "BUIA_PASSWORD in .env is empty, so nobody can log in. Put a password after the = and run this again."
}

# 5c. Session-signing salt: random and stable for this installation, added on an update too (the agent signs sessions with a key derived from the password and this salt)
if (-not (@(Get-Content .env) -match '^BUIA_SESSION_SALT=.+')) {
    Set-EnvValue 'BUIA_SESSION_SALT' (New-Password)
}

# 6. Folders and memory file the containers mount. As in docker-compose.yml, BUIA_PATH defaults to the folder above core and VAULT_PATH to .vault inside it
$buia = Get-EnvValue 'BUIA_PATH'
if (-not $buia) { $buia = (Resolve-Path '..').Path }
$vault = Get-EnvValue 'VAULT_PATH'
if (-not $vault) { $vault = Join-Path $buia '.vault' }
New-Item -ItemType Directory -Force -Path $buia, $vault | Out-Null
if (-not (Test-Path "$buia/BUIA.md")) {
    Copy-Item ../templates/BUIA.md "$buia/BUIA.md"
    Write-Host "Created $buia/BUIA.md from the template"
}

# 7. Optional: image/video generation sources and model folder. The sd service is rebuilt whenever sdcpp\ exists, so an update keeps it.
$compose = @()
if ($Sd -and -not (Test-Path sdcpp)) {
    git -c advice.detachedHead=false clone --branch $SdTag --depth 1 --recurse-submodules --shallow-submodules https://github.com/leejet/stable-diffusion.cpp sdcpp
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -Recurse -Force sdcpp -ErrorAction SilentlyContinue
        throw "git clone of stable-diffusion.cpp failed"
    }
}
if (Test-Path sdcpp) {
    New-Item -ItemType Directory -Force -Path sdmodels | Out-Null
    $compose = @("--profile", "sd")
    # the sd image is compiled for one GPU architecture (docker-compose.yml: SD_CUDA_ARCHITECTURES, default 120 = RTX 50 series)
    if (-not (@(Get-Content .env) -match '^SD_CUDA_ARCHITECTURES=')) {
        $cap = ''
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $cap = ([string](nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)) -replace '[^0-9]', '' }
        if ($cap) {
            Set-EnvValue 'SD_CUDA_ARCHITECTURES' $cap
            Write-Host "Set SD_CUDA_ARCHITECTURES=$cap in .env (your GPU's compute capability)."
        } else {
            Write-Warning "Could not read the GPU's compute capability: the sd image is built for 120 (RTX 50 series). Set SD_CUDA_ARCHITECTURES in .env if yours differs."
        }
    }
    if ($Sd) {
        Write-Host "Download an image and a video model in the web UI (Settings > Image and video models), or put the files named in SD_ARGS / SD_VIDEO_ARGS (.env) into $PSScriptRoot\sdmodels"
    }
}

# 7b. Optional: CogVideoX-2b. The flag lives in .env so that the next update keeps rebuilding the service
if ($CogVideo) { Set-EnvValue 'COGVIDEO' 'on' }
if (@(Get-Content .env) -match '^COGVIDEO=on') {
    New-Item -ItemType Directory -Force -Path sdmodels | Out-Null
    $compose += @("--profile", "cogvideo")
    if ($CogVideo) {
        Write-Host "Download CogVideoX 2B in the web UI (Settings > Image and video models), then click Use. See README > CogVideoX before the first video."
    }
}

# 8. Optional: what the newest llama.cpp image is now (the compose file keeps the pinned one)
if ($Llama) {
    $image = $env:LLAMA_IMAGE
    if (-not $image) { $image = Get-EnvValue 'LLAMA_IMAGE' }
    if (-not $image) { $image = $LlamaDefault }
    $latest = $image -replace '@.*$', ''
    docker pull $latest
    Check "docker pull $latest"
    Write-Host "Newest $latest is:"
    docker image inspect $latest --format "{{index .RepoDigests 0}}"
}

# 9. Build and start
docker compose @compose up -d --build
Check "docker compose up"

# 10. Wait until llama-server answers (up to 2 minutes)
Write-Host "Waiting for BUIA..."
$ready = $false
$wait = [Diagnostics.Stopwatch]::StartNew()
while ($wait.Elapsed.TotalSeconds -lt 120 -and -not $ready) {
    $remaining = [math]::Max(1, [math]::Ceiling(120 - $wait.Elapsed.TotalSeconds))
    try { $ready = [bool](Invoke-RestMethod http://127.0.0.1:8000/api/health -TimeoutSec ([math]::Min(3, $remaining))).ready } catch { }
    if (-not $ready -and $wait.Elapsed.TotalSeconds -lt 120) { Start-Sleep -Seconds ([math]::Min(2, [math]::Max(1, [math]::Floor(120 - $wait.Elapsed.TotalSeconds)))) }
}
if (-not $ready) {
    Write-Warning "BUIA is not ready after 120 seconds (the model may still be loading). Current services and recent logs:"
    docker compose @compose ps
    docker compose @compose logs --tail 50 llama agent
}

# 11. Optional: phone access over Tailscale
if ($Tailscale) {
    tailscale serve --bg 8000
    Check "tailscale serve"
    tailscale serve status
}

Write-Host ""
if ($mode -eq 'update') {
    docker compose @compose ps
    Write-Host "Updated. If a release note says new settings were added, compare core\.env with core\.env.example."
}
Write-Host "BUIA is at http://127.0.0.1:8000  -  open Settings to download and select a model."
