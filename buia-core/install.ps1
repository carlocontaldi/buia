# Installs or re-applies BUIA. Run .\install.cmd in the folder above (it calls this script with the execution policy bypassed), or .\install.ps1 from this folder.
# Add -Tailscale to also expose the UI to your tailnet (phone access).
# Add -Sd to also set up image/video generation (clones stable-diffusion.cpp and builds the sd service; the first build is long).
# Add -CogVideo to also build the cogvideo service (CogVideoX-2b text-to-video, PyTorch; the first build downloads several GB).
param([switch]$Tailscale, [switch]$Sd, [switch]$CogVideo)

$SdTag = "master-920-2f88688"   # stable-diffusion.cpp release the sd service is built from

Set-Location $PSScriptRoot

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function New-Password {   # 32 hex characters from the OS random number generator (hex: no $ or # for Compose to trip over)
    $b = New-Object byte[] 16
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
    -join ($b | ForEach-Object { $_.ToString('x2') })
}

# 1. Docker must be running
docker info *> $null
Check "Docker (is Docker Desktop running?)"

# 2. Warn early if containers cannot see the GPU
docker run --rm --gpus all ubuntu nvidia-smi *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Docker cannot see the NVIDIA GPU. Check the NVIDIA driver and Docker Desktop's WSL2 backend before continuing."
}

# 3. .env: created once from .env.example, never overwritten
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    $parent = (Resolve-Path "..").Path -replace '\\', '/'
    (Get-Content .env) `
        -replace '^BUIA_PATH=.*', "BUIA_PATH=$parent" `
        -replace '^VAULT_PATH=.*', "VAULT_PATH=$parent/.obsidian" |
        Set-Content .env -Encoding ascii
    Write-Host "Created .env with BUIA_PATH=$parent"
}

# 3b. Web UI password: a random one the first time (fresh .env, or an .env from before the login existed), never overwritten
$envLines = @(Get-Content .env)
if (-not ($envLines -match '^BUIA_PASSWORD=')) {
    $pw = New-Password
    Add-Content .env "`nBUIA_PASSWORD=$pw" -Encoding ascii
    Write-Host "Web UI password: $pw   (saved in .env as BUIA_PASSWORD; change it there)"
} elseif (-not ($envLines -match '^BUIA_PASSWORD=.+')) {
    Write-Warning "BUIA_PASSWORD in .env is empty, so nobody can log in. Put a password after the = and run this again."
}

# 4. Folders and memory file the containers mount
$buia  = ((Get-Content .env) | Where-Object { $_ -match '^BUIA_PATH=' }) -replace '^BUIA_PATH=', ''
$vault = ((Get-Content .env) | Where-Object { $_ -match '^VAULT_PATH=' }) -replace '^VAULT_PATH=', ''
New-Item -ItemType Directory -Force -Path $buia, $vault | Out-Null
if (-not (Test-Path "$buia/BUIA.md")) {
    Copy-Item ../templates/BUIA.md "$buia/BUIA.md"
    Write-Host "Created $buia/BUIA.md from the template"
}

# 5. Optional: image/video generation sources and model folder
$compose = @()
if ($Sd) {
    if (-not (Test-Path sdcpp)) {
        git -c advice.detachedHead=false clone --branch $SdTag --depth 1 --recurse-submodules --shallow-submodules https://github.com/leejet/stable-diffusion.cpp sdcpp
        if ($LASTEXITCODE -ne 0) {
            Remove-Item -Recurse -Force sdcpp -ErrorAction SilentlyContinue
            throw "git clone of stable-diffusion.cpp failed"
        }
    }
    New-Item -ItemType Directory -Force -Path sdmodels | Out-Null
    $compose = @("--profile", "sd")
    Write-Host "Download an image and a video model in the web UI (Models > Image and video models), or put the files named in SD_ARGS / SD_VIDEO_ARGS (.env) into $PSScriptRoot\sdmodels"
}

# 5b. Optional: CogVideoX-2b. The flag lives in .env so that update.ps1 keeps rebuilding the service
if ($CogVideo) {
    Set-Content .env (@(Get-Content .env | Where-Object { $_ -notmatch '^COGVIDEO=' }) + 'COGVIDEO=on') -Encoding ascii
}
if (@(Get-Content .env) -match '^COGVIDEO=on') {
    New-Item -ItemType Directory -Force -Path sdmodels | Out-Null
    $compose += @("--profile", "cogvideo")
    Write-Host "Download CogVideoX 2B in the web UI (Models > Image and video models), then click Use. See README > CogVideoX before the first video."
}

# 6. Build and start
docker compose @compose up -d --build
Check "docker compose up"

# 7. Wait until llama-server answers (up to 2 minutes)
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

# 8. Optional: phone access over Tailscale
if ($Tailscale) {
    tailscale serve --bg 8000
    Check "tailscale serve"
    tailscale serve status
}

Write-Host ""
Write-Host "BUIA is at http://127.0.0.1:8000  -  open Models to download and select a model."
