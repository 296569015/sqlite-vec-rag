# Download SQLite3 Amalgamation Script
# Usage: .\download_sqlite3.ps1

param(
    [string]$Version = "3450200",
    [string]$OutputDir = "build/sqlite3"
)

Write-Host "Downloading SQLite3 Amalgamation..." -ForegroundColor Green

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Download URL
$Url = "https://www.sqlite.org/2024/sqlite-amalgamation-$Version.zip"
$ZipFile = "$OutputDir\sqlite3.zip"

try {
    # Download
    Write-Host "Downloading from $Url..." -NoNewline
    Invoke-WebRequest -Uri $Url -OutFile $ZipFile -TimeoutSec 120
    Write-Host " OK" -ForegroundColor Green
    
    # Extract
    Write-Host "Extracting..." -NoNewline
    Expand-Archive -Path $ZipFile -DestinationPath $OutputDir -Force
    
    # Move files from subdirectory
    $SubDir = Get-ChildItem -Path $OutputDir -Directory | Select-Object -First 1
    if ($SubDir) {
        Move-Item -Path "$($SubDir.FullName)\sqlite3.c" -Destination $OutputDir -Force
        Move-Item -Path "$($SubDir.FullName)\sqlite3.h" -Destination $OutputDir -Force
        Remove-Item -Path $SubDir.FullName -Recurse -Force
    }
    
    Remove-Item -Path $ZipFile -Force
    Write-Host " OK" -ForegroundColor Green
    
    Write-Host "`nSQLite3 files downloaded to: $(Resolve-Path $OutputDir)" -ForegroundColor Green
    Write-Host "Files: sqlite3.c, sqlite3.h" -ForegroundColor Gray
    
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. Reconfigure with: cmake .. -DUSE_SQLITE3=ON" -ForegroundColor Gray
    Write-Host "  2. Rebuild with: cmake --build . --config Release" -ForegroundColor Gray
    
} catch {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}
