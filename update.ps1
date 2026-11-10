# Updates BUIA from the repo. Run from this folder:  .\update.cmd   (it calls this script with the execution policy bypassed, like install.cmd)
# Add -Llama to also look at the newest llama.cpp server image. docker-compose.yml pins the image by digest (a known working build),
# so this only shows what the moving tag points to now. To adopt it, paste its digest into docker-compose.yml and run this script again.
param([switch]$Llama)

Set-Location $PSScriptRoot

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

function New-Password {   # 32 hex characters from the OS random number generator (hex: no $ or # for Compose to trip over)
    $b = New-Object byte[] 16
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b)
    -join ($b | ForEach-Object { $_.ToString('x2') })
}

git pull --ff-only
Check "git pull"

# the web UI login came after the first release: an .env from before it has no password yet (install.ps1 does the same for a fresh one)
if ((Test-Path .env) -and -not (@(Get-Content .env) -match '^BUIA_PASSWORD=')) {
    $pw = New-Password
    Add-Content .env "`nBUIA_PASSWORD=$pw" -Encoding ascii
    Write-Host "Web UI password: $pw   (saved in .env as BUIA_PASSWORD; change it there)"
}

# image/video generation is rebuilt too if it was installed (install.ps1 -Sd)
$compose = @()
if (Test-Path sdcpp) { $compose = @("--profile", "sd") }

if ($Llama) {
    $latest = "ghcr.io/ggml-org/llama.cpp:server-cuda13"   # keep equal to the tag in docker-compose.yml
    docker pull $latest
    Check "docker pull $latest"
    Write-Host "Newest $latest is:"
    docker image inspect $latest --format "{{index .RepoDigests 0}}"
}

docker compose @compose up -d --build
Check "docker compose up"

docker compose @compose ps
Write-Host ""
Write-Host "Updated. If a release note says new settings were added, compare your .env with .env.example."
