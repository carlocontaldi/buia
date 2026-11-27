# Updates BUIA from the repo. Run .\update.cmd in the folder above (it calls this script with the execution policy bypassed, like install.cmd).
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

# an .env without BUIA_PASSWORD leaves nobody able to log in: add a random one (install.ps1 does the same)
if ((Test-Path .env) -and -not (@(Get-Content .env) -match '^BUIA_PASSWORD=')) {
    $pw = New-Password
    Add-Content .env "`nBUIA_PASSWORD=$pw" -Encoding ascii
    Write-Host "Web UI password: $pw   (saved in .env as BUIA_PASSWORD; change it there)"
}

# Give existing installations a stable random salt for session signing.
if ((Test-Path .env) -and -not (@(Get-Content .env) -match '^BUIA_SESSION_SALT=.+')) {
    $salt = New-Password
    Add-Content .env "`nBUIA_SESSION_SALT=$salt" -Encoding ascii
}

# image/video generation is rebuilt too if it was installed (install.ps1 -Sd)
$compose = @()
if (Test-Path sdcpp) { $compose = @("--profile", "sd") }
# and the CogVideoX service, if install.ps1 -CogVideo turned it on in .env
if ((Test-Path .env) -and (@(Get-Content .env) -match '^COGVIDEO=on')) { $compose += @("--profile", "cogvideo") }

if ($Llama) {
    $latest = "ghcr.io/ggml-org/llama.cpp:server-cuda13"   # keep equal to the tag in docker-compose.yml
    docker pull $latest
    Check "docker pull $latest"
    Write-Host "Newest $latest is:"
    docker image inspect $latest --format "{{index .RepoDigests 0}}"
}

docker compose @compose up -d --build
Check "docker compose up"

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

docker compose @compose ps
Write-Host ""
Write-Host "Updated. If a release note says new settings were added, compare your .env with .env.example."
