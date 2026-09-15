$platformLocal = "$env:LOCALAPPDATA\Arduino15\packages\esp32\hardware\esp32\3.2.1\platform.local.txt"
$content = @"
# Workaround for ESP32 core 3.2.1 linker errors where core symbols such as
# Print, IPAddress, HardwareSerial, and Serial0 are reported as undefined.
compiler.ar.flags=crs
"@

Set-Content -Path $platformLocal -Value $content -Encoding ASCII
Write-Host "Wrote $platformLocal"
Write-Host "Restart Arduino IDE, then compile/upload again."
