# 下载 sqlite-vec 扩展脚本
# 用于 Windows 平台

param(
    [string]$Version = "v0.1.6",
    [string]$OutputDir = "third_party"
)

Write-Host "Downloading sqlite-vec $Version for Windows..." -ForegroundColor Green

# 创建输出目录
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# 下载地址
$BaseUrl = "https://github.com/asg017/sqlite-vec/releases/download/$Version"
$Files = @(
    "sqlite-vec-$Version-loadable-windows-x64.dll",
    "sqlite-vec-$Version-loadable-windows-x64.vec0.dll"
)

foreach ($File in $Files) {
    $Url = "$BaseUrl/$File"
    $OutputPath = "$OutputDir/$File"
    
    Write-Host "Downloading $File..." -NoNewline
    
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath -ErrorAction Stop
        Write-Host " OK" -ForegroundColor Green
        
        # 同时创建一个简化名称的副本
        if ($File -like "*vec0.dll") {
            $SimpleName = "$OutputDir/vec0.dll"
            Copy-Item $OutputPath $SimpleName -Force
            Write-Host "  -> Copied to vec0.dll" -ForegroundColor Gray
        }
    } catch {
        Write-Host " Failed" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
    }
}

Write-Host "`nDownload complete. Files saved to: $(Resolve-Path $OutputDir)" -ForegroundColor Green
Write-Host "`nUsage in your code:" -ForegroundColor Cyan
Write-Host "  1. Copy vec0.dll to your executable directory" -ForegroundColor Gray
Write-Host "  2. The library will automatically load the extension" -ForegroundColor Gray
