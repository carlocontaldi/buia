# Installs or re-applies BUIA. Run from this folder:  .\install.ps1
# Add -Tailscale to also expose the UI to your tailnet (phone access).
# Add -Sd to also set up image/video generation (clones stable-diffusion.cpp and builds the sd service; the first build is long).
param([switch]$Tailscale, [switch]$Sd)

$SdTag = "master-920-2f88688"   # stable-diffusion.cpp release the sd service is built from

Set-Location $PSScriptRoot

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
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

# 4. Folders and memory file the containers mount
$buia  = ((Get-Content .env) | Where-Object { $_ -match '^BUIA_PATH=' }) -replace '^BUIA_PATH=', ''
$vault = ((Get-Content .env) | Where-Object { $_ -match '^VAULT_PATH=' }) -replace '^VAULT_PATH=', ''
New-Item -ItemType Directory -Force -Path $buia, $vault | Out-Null
if (-not (Test-Path "$buia/BUIA.md")) {
    Copy-Item templates/BUIA.md "$buia/BUIA.md"
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
    Write-Host "Put the model files named in SD_ARGS (.env) into $PSScriptRoot\sdmodels"
}

# 6. Build and start
docker compose @compose up -d --build
Check "docker compose up"

# 7. Wait until llama-server answers (up to 2 minutes)
Write-Host "Waiting for BUIA..."
$ready = $false
for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
    try { $ready = [bool](Invoke-RestMethod http://127.0.0.1:8000/api/health -TimeoutSec 3).ready } catch { }
    if (-not $ready) { Start-Sleep -Seconds 2 }
}
if (-not $ready) {
    Write-Warning "Not ready yet. Look at: docker compose logs llama --tail 50"
}

# 8. Optional: phone access over Tailscale
if ($Tailscale) {
    tailscale serve --bg 8000
    Check "tailscale serve"
    tailscale serve status
}

Write-Host ""
Write-Host "BUIA is at http://127.0.0.1:8000  -  open Models to download and select a model."
