# Run autoencoder training; restart on failure until exit 0.
$maxRestarts = 20
$restarts = 0
$logDir = "result_logs"
$stderrLog = "figures\autoencoder\ae_stderr.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path "figures\autoencoder" | Out-Null

while ($true) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting autoencoder training (attempt $($restarts + 1))..."
    $proc = Start-Process -FilePath "python" -ArgumentList "synapse_autoencoder.py --epochs 30 --max_samples 1500 --batch_size 64 --checkpoint_every 10" -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow -Wait -RedirectStandardOutput "$logDir\ae_stdout.log" -RedirectStandardError $stderrLog
    if ($proc.ExitCode -eq 0) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Training finished successfully."
        break
    }
    $restarts++
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Process exited with code $($proc.ExitCode). Check $stderrLog and figures\autoencoder\run_error.log"
    if (Test-Path "figures\autoencoder\run_error.log") { Get-Content "figures\autoencoder\run_error.log" -Tail 30 }
    if ($restarts -ge $maxRestarts) {
        Write-Host "Max restarts ($maxRestarts) reached. Stopping."
        exit 1
    }
    Write-Host "Restarting in 15 seconds..."
    Start-Sleep -Seconds 15
}
