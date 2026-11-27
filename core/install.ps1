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
    Write-Host "Created .env"
}

# 3b. Web UI password: a random one if .env has none, never overwritten
$envLines = @(Get-Content .env)
if (-not ($envLines -match '^BUIA_PASSWORD=')) {
    $pw = New-Password
    Add-Content .env "`nBUIA_PASSWORD=$pw" -Encoding ascii
    Write-Host "Web UI password: $pw   (saved in .env as BUIA_PASSWORD; change it there)"
} elseif (-not ($envLines -match '^BUIA_PASSWORD=.+')) {
    Write-Warning "BUIA_PASSWORD in .env is empty, so nobody can log in. Put a password after the = and run this again."
}

# 3c. Session-signing salt: random and stable for this installation
$envLines = @(Get-Content .env)
if (-not ($envLines -match '^BUIA_SESSION_SALT=.+')) {
    $salt = New-Password
    $utf8 = New-Object Text.UTF8Encoding($false)
    $envPath = Join-Path $PSScriptRoot '.env'
    $text = [IO.File]::ReadAllText($envPath, $utf8)
    if ($text -match '(?m)^BUIA_SESSION_SALT=') {
        $new = $text -replace '(?m)^BUIA_SESSION_SALT=[^\r\n]*', "BUIA_SESSION_SALT=$salt"
    } else {
        $new = $text.TrimEnd() + "`nBUIA_SESSION_SALT=$salt`n"
    }
    [IO.File]::WriteAllText($envPath, $new, $utf8)
}

# 4. Folders and memory file the containers mount. As in docker-compose.yml, BUIA_PATH defaults to the folder above core and VAULT_PATH to .vault inside it
function Get-EnvValue($name) { (@(Get-Content .env -Encoding UTF8) | Where-Object { $_ -match "^$name=\S" } | Select-Object -First 1) -replace "^$name=", '' }
$buia = Get-EnvValue 'BUIA_PATH'
if (-not $buia) { $buia = (Resolve-Path '..').Path }
$vault = Get-EnvValue 'VAULT_PATH'
if (-not $vault) { $vault = Join-Path $buia '.vault' }
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
    Write-Host "Download an image and a video model in the web UI (Settings > Image and video models), or put the files named in SD_ARGS / SD_VIDEO_ARGS (.env) into $PSScriptRoot\sdmodels"
}

# 5b. Optional: CogVideoX-2b. The flag lives in .env so that update.ps1 keeps rebuilding the service
if ($CogVideo) {
    # Only the COGVIDEO line changes. .env is read and written as UTF-8, which is what Compose reads: Set-Content -Encoding ascii turned every
    # accented character (in a path, say) into "?", and Windows PowerShell 5.1 reads a file without a BOM as ANSI.
    $utf8 = New-Object Text.UTF8Encoding($false)
    $envPath = Join-Path $PSScriptRoot '.env'
    $text = [IO.File]::ReadAllText($envPath, $utf8)
    if ($text -match '(?m)^COGVIDEO=') { $new = $text -replace '(?m)^COGVIDEO=[^\r\n]*', 'COGVIDEO=on' } else { $new = $text.TrimEnd() + "`nCOGVIDEO=on`n" }
    if ($new -ne $text) { [IO.File]::WriteAllText($envPath, $new, $utf8) }
}
if (@(Get-Content .env) -match '^COGVIDEO=on') {
    New-Item -ItemType Directory -Force -Path sdmodels | Out-Null
    $compose += @("--profile", "cogvideo")
    Write-Host "Download CogVideoX 2B in the web UI (Settings > Image and video models), then click Use. See README > CogVideoX before the first video."
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
Write-Host "BUIA is at http://127.0.0.1:8000  -  open Settings to download and select a model."
