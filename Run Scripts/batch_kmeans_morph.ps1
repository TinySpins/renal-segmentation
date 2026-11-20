# batch_kmeans_morph.ps1
# Batch renal segmentation (K-means + morphology) via Docker

$ErrorActionPreference = "Stop"

# --- configure these ---
$DATA = "E:\Renal Segmentation\data"                               # host folder mounted as /data
$SRC  = "E:\Renal Segmentation\code\renal-segmentation"            # contains Source\ and renalSegment_kmeans_morph.py
$InRoot  = Join-Path $DATA "DicomIn"                               # each subfolder = one dataset
$OutRoot = Join-Path $DATA "DicomOut_kmeans_morph"
$Image   = "umityoruk/renal-segmentation"

# Extra args for the Python CLI (optional)
$ExtraArgs = @(   							# e.g. @("--moco") or @("--gap-factor", "2.5")
	"--hilum-open-fraction", "0.4"          # enable hilum cleanup; open radius = 0.4 * minor-axis (mm)
	"--hilum-si-band", "0.40"
)
# examples:
# $ExtraArgs = @()   # all defaults (hilum cleanup OFF, no MoCo)
# $ExtraArgs = @("--hilum-open-fraction", "0.4")   # enable hilum cleanup; open radius = 0.4 * minor-axis (mm), maximum is 0.5
# $ExtraArgs = @("--hilum-open-fraction", "0.4", "--hilum-si-band", "0.40")  # hilum on; central 40% SI band gates removals
# $ExtraArgs = @("--gap-factor", "2.5")   # trailing trigger gap vs median to drop last phases
# $ExtraArgs = @("--threshold-multiplier", "1.0")   # GrabCut threshold multiplier
# $ExtraArgs = @("--moco")   # enable motion correction
# $ExtraArgs = @("--moco", "--moco-ref-time", "10")   # MoCo with 10 s reference time
# $ExtraArgs = @("--moco", "--moco-ref-time", "10", "--moco-maxiter", "40")   # MoCo + optimizer iterations
# $ExtraArgs = @("--moco", "--second-grabcut")   # MoCo then re-run GrabCut


# --- helpers ---
function Get-DicomLeafDir {
    param([string]$Root)

    function Test-IsBadDicomsDir {
        param([string]$Path)
        $parts = $Path -split '[\\/]'
        return ($parts -contains '_bad_dicom')
    }

    function Test-HasDicomFiles {
        param([string]$Dir)
        if (Test-IsBadDicomsDir $Dir) { return $false }
        $n = (Get-ChildItem -Path $Dir -File -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Extension -match '\.(dcm|mag)$' -and
                $_.DirectoryName -notmatch '\\_bad_dicom($|\\)'
            }).Count
        return ($n -gt 0)
    }

    $dirs = Get-ChildItem -Path $Root -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            -not (Test-IsBadDicomsDir $_.FullName) -and
            (Test-HasDicomFiles $_.FullName)
        }

    if (Test-HasDicomFiles $Root) {
        $dirs = @([pscustomobject]@{ FullName = $Root }) + @($dirs)
    }

    if (-not $dirs -or $dirs.Count -eq 0) {
        return $null
    }

    # Deepest folder that still has readable series DICOMs (not quarantine)
    $leaf = $dirs |
        Sort-Object @{ Expression = { ($_.FullName -split '[\\/]').Count } },
                    @{ Expression = { $_.FullName.Length } } -Descending |
        Select-Object -First 1

    return $leaf.FullName
}

function ConvertTo-ContainerPath {
    param([string]$HostPath, [string]$HostDataRoot)
    # Map host $DATA\... -> /data/...
    $full = [System.IO.Path]::GetFullPath($HostPath)
    $root = [System.IO.Path]::GetFullPath($HostDataRoot)
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path not under DATA root: $full"
    }
    $rel = $full.Substring($root.Length).TrimStart('\', '/')
    $unix = ($rel -replace '\\', '/')
    if ([string]::IsNullOrEmpty($unix)) { return "/data" }
    return "/data/$unix"
}

# --- main ---
if (-not (Test-Path $InRoot)) { throw "DicomIn not found: $InRoot" }
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$datasets = Get-ChildItem -Path $InRoot -Directory
if ($datasets.Count -eq 0) {
    Write-Host "No dataset folders under $InRoot"
    exit 0
}

Write-Host "Found $($datasets.Count) dataset folder(s) under $InRoot"

foreach ($ds in $datasets) {
    $name = $ds.Name
    Write-Host ""
    Write-Host "========== $name =========="

    $leaf = Get-DicomLeafDir -Root $ds.FullName
    if (-not $leaf) {
        Write-Host "SKIP: no .dcm/.mag found under $($ds.FullName)"
        continue
    }

    Write-Host "DICOM leaf: $leaf"

    $outHost = Join-Path $OutRoot $name
    New-Item -ItemType Directory -Force -Path $outHost | Out-Null

    $inCont  = ConvertTo-ContainerPath -HostPath $leaf    -HostDataRoot $DATA
    $outCont = ConvertTo-ContainerPath -HostPath $outHost -HostDataRoot $DATA

    Write-Host "Container in:  $inCont"
    Write-Host "Container out: $outCont"

    docker run --rm `
      -v "${DATA}:/data" `
      -v "${SRC}\Source:/renal-segmentation/Source" `
      -v "${SRC}\renalSegment_kmeans_morph.py:/renal-segmentation/renalSegment_kmeans_morph.py" `
      $Image `
      python /renal-segmentation/renalSegment_kmeans_morph.py `
        $inCont $outCont @ExtraArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $name (exit $LASTEXITCODE)"
    } else {
        Write-Host "OK: $name -> $outHost"
    }
}

Write-Host ""
Write-Host "Batch finished. Outputs under $OutRoot"