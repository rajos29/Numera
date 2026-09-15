# ESP32 Arithmetic Serial Backend

This is the first ESP32 backend for the arithmetic characterization project.

The uploadable Arduino sketch is:

```text
firmware_esp32/arithmetic_serial/arithmetic_serial.ino
```

## Upload

1. Open `firmware_esp32/arithmetic_serial/arithmetic_serial.ino` in Arduino IDE.
2. Select the ESP32 board profile that matches the connected board.
3. Select the serial port.
4. Upload.
5. Open Serial Monitor at `115200` baud.
6. Set line ending to `Newline` or `Both NL & CR`.

## Profile Request

```json
{"op":"profile"}
```

From PowerShell:

```powershell
python python_lab/esp32_serial_runner.py --port COM3 --profile
```

The board responds with chip model, core count, CPU frequency, heap, and
`sizeof(double)`.

## Benchmark Request

```json
{"op":"divide","numeric_type":"float32","samples":1000,"passes":1000,"seed":20260912}
```

From PowerShell:

```powershell
python python_lab/esp32_serial_runner.py --port COM3 --operation divide --numeric-type float32 --samples 1000 --passes 1000
```

Supported operations:

- `suite`
- `empty_loop`
- `add`
- `subtract`
- `multiply`
- `divide`

Supported numeric types:

- `float32`
- `float64`

## Current Methodology

The ESP32 generates deterministic input pairs on-device from the supplied seed.

The timed region repeats `passes` over `samples`, so:

```text
measured_evaluations = samples * passes
```

The response reports:

- total runtime in microseconds
- runtime per evaluation in nanoseconds
- bounded result checksum
- heap before and after
- minimum free heap before and after

This is still a first backend. It uses `micros()` wall-clock timing and does not
yet collect CPU cycles.

## Suite Request

To run every current arithmetic combination from Serial Monitor:

```json
{"op":"suite","samples":1000,"passes":1000,"seed":20260912}
```

The board emits:

- one `suite_start` row
- one result row for each operation and numeric type
- one `suite_end` row

From PowerShell, collect the same suite into CSV:

```powershell
python python_lab/esp32_serial_runner.py --port COM3 --trials 5
```

No arguments runs the full suite once by default. Use `--trials 5` for the
recommended repeated baseline. Use `--single` when you only want one operation
for debugging:

```powershell
python python_lab/esp32_serial_runner.py --port COM3 --single --operation divide --numeric-type float32 --samples 1000 --passes 1000
```

## Report Generation

After collecting a suite CSV, generate the ESP32 report and figures:

```powershell
python python_lab/esp32_report.py
```

By default this reads the newest `*_suite.csv` under `python_lab/data/esp32/`.
Use `--csv path\to\suite.csv` to report a specific run.
