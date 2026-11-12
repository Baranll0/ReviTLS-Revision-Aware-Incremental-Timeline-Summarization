<#
.SYNOPSIS
    Downloads T17, Crisis (from news-tls Google Drive), and Open-TLS (from Alibaba-NLP/CHRONOS).
#>
param(
    [string]$RawRoot = "data\raw"
)

$ErrorActionPreference = "Stop"

function Get-DriveSubfolders {
    param([string]$FolderId)
    $html = curl.exe -sL --max-time 30 "https://drive.google.com/drive/folders/$FolderId"
    $pattern = '\[null,"([A-Za-z0-9_-]{28,45})"\],null,null,null,"application/vnd\.google-apps\.folder".{0,600}?\["([^"]{2,40})"'
    $items = @{}
    foreach ($m in [regex]::Matches($html, $pattern)) {
        $id = $m.Groups[1].Value
        $name = $m.Groups[2].Value
        if (-not $items.ContainsKey($name)) {
            $items[$name] = $id
        }
    }
    return $items
}

function Get-DriveFiles {
    param([string]$FolderId)
    $html = curl.exe -sL --max-time 30 "https://drive.google.com/drive/folders/$FolderId"
    $pattern = '\[null,"([A-Za-z0-9_-]{28,45})"\],null,null,null,"[^"]+?".{0,600}?\["([^"]+\.(?:json|jsonl|gz))"'
    $items = @{}
    foreach ($m in [regex]::Matches($html, $pattern)) {
        $id = $m.Groups[1].Value
        $name = $m.Groups[2].Value
        if (-not $items.ContainsKey($name)) {
            $items[$name] = $id
        }
    }
    return $items
}

function Download-DriveDataset {
    param(
        [string]$DatasetName,
        [string]$FolderId
    )
    Write-Host "=== Downloading $DatasetName ===" -ForegroundColor Cyan
    $topics = Get-DriveSubfolders -FolderId $FolderId
    foreach ($topic in ($topics.Keys | Sort-Object)) {
        $topicFolderId = $topics[$topic]
        $targetDir = Join-Path $RawRoot "$DatasetName\$topic"
        New-Item -ItemType Directory -Force $targetDir | Out-Null

        $files = Get-DriveFiles -FolderId $topicFolderId
        foreach ($fname in ($files.Keys | Sort-Object)) {
            $fid = $files[$fname]
            $outPath = Join-Path $targetDir $fname
            if (Test-Path $outPath) {
                Write-Host "  [skip] $DatasetName/$topic/$fname"
                continue
            }
            Write-Host "  [get ] $DatasetName/$topic/$fname"
            curl.exe -fL --retry 3 --max-time 180 -o $outPath "https://drive.usercontent.google.com/download?id=$fid&export=download&confirm=t"
        }
    }
}

# Google Drive root folders from complementizer/news-tls
Download-DriveDataset -DatasetName "t17" -FolderId "10X1Hd1cWPebTzSkl6fyyzO9hHmO-OxkK"
Download-DriveDataset -DatasetName "crisis" -FolderId "1PhH0Z1MZbmvt-DGDo-51YGW0KCQ4UZYv"

# Open-TLS from Alibaba-NLP/CHRONOS
$chronosDir = Join-Path $RawRoot "chronos_repo\CHRONOS-main"
if (-not (Test-Path $chronosDir)) {
    Write-Host "=== Downloading CHRONOS (Open-TLS) ===" -ForegroundColor Cyan
    $zipPath = Join-Path $RawRoot "chronos.zip"
    curl.exe -sL --max-time 60 -o $zipPath "https://github.com/Alibaba-NLP/CHRONOS/archive/refs/heads/main.zip"
    Expand-Archive -Force $zipPath (Join-Path $RawRoot "chronos_repo")
    Remove-Item $zipPath
}

Write-Host "Done! Raw datasets are under $RawRoot" -ForegroundColor Green
