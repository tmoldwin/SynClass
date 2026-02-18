# Kill any existing training processes, then run once and monitor (restart on crash).
$ErrorActionPreference = "Continue"
$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location $scriptDir

# Kill only our training processes
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='python3.13.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -like '*synapse_autoencoder*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

$csvPath = "figures\autoencoder\training_outputs.csv"
$errorLog = "figures\autoencoder\run_error.log"
$targetEpochs = 30
$checkIntervalSec = 45
$maxRestarts = 5
$restarts = 0

function Start-Training {
  $proc = Start-Process -FilePath "python" -ArgumentList "synapse_autoencoder.py --epochs $targetEpochs --max_samples 1500 --batch_size 64 --checkpoint_every 10" -WorkingDirectory $scriptDir -PassThru -NoNewWindow
  return $proc
}

function Get-CsvEpochCount {
  if (-not (Test-Path $csvPath)) { return 0 }
  $lines = Get-Content $csvPath -ErrorAction SilentlyContinue
  return [Math]::Max(0, $lines.Count - 1)
}

while ($restarts -le $maxRestarts) {
  Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting training (restart $restarts)..."
  $p = Start-Training
  $lastEpochs = 0

  while (-not $p.HasExited) {
    Start-Sleep -Seconds $checkIntervalSec
    $n = Get-CsvEpochCount
    if ($n -gt $lastEpochs) {
      Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Epochs completed: $n / $targetEpochs"
      $lastEpochs = $n
    }
  }

  $exitCode = $p.ExitCode
  $finalEpochs = Get-CsvEpochCount
  Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Process exited with code $exitCode. Epochs in CSV: $finalEpochs"

  if ($exitCode -eq 0 -or $finalEpochs -ge $targetEpochs) {
    Write-Host "Run finished successfully."
    exit 0
  }

  if (Test-Path $errorLog) {
    Write-Host "Last 25 lines of run_error.log:"
    Get-Content $errorLog -Tail 25
  }
  $restarts++
  if ($restarts -gt $maxRestarts) {
    Write-Host "Max restarts reached. Stopping."
    exit 1
  }
  Write-Host "Restarting in 15 seconds..."
  Start-Sleep -Seconds 15
}
