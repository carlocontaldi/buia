# Updates BUIA from the repo. Run from this folder:  .\update.ps1
# Add -Llama to also pull the newest llama.cpp server image (the tag in docker-compose.yml moves over time).
param([switch]$Llama)

Set-Location $PSScriptRoot

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

git pull --ff-only
Check "git pull"

# image/video generation is rebuilt too if it was installed (install.ps1 -Sd)
$compose = @()
if (Test-Path sdcpp) { $compose = @("--profile", "sd") }

if ($Llama) {
    docker compose @compose pull llama
    Check "docker compose pull llama"
}

docker compose @compose up -d --build
Check "docker compose up"

docker compose @compose ps
Write-Host ""
Write-Host "Updated. If a release note says new settings were added, compare your .env with .env.example."
