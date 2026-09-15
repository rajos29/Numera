/*
  Workstation/ESP32 arithmetic characterization prototype.

  Upload this sketch with Arduino IDE, open Serial Monitor at 115200 baud, and
  send one JSON-ish request per line, for example:

  {"op":"divide","numeric_type":"float32","samples":1000,"passes":1000,"seed":20260912}

  Supported operations:
    profile
    suite
    empty_loop
    add
    subtract
    multiply
    divide

  Supported numeric types:
    float32
    float64

  This is a first ESP32 backend, not final benchmark methodology.
*/

#include <Arduino.h>
#include <math.h>
#include <stdint.h>

static const uint32_t BAUD_RATE = 115200;
static const size_t MAX_LINE = 256;

static volatile double g_result_sink = 0.0;
static char g_line[MAX_LINE];
static size_t g_line_len = 0;
static const float CHECKSUM_BOUND_FLOAT32 = 1000000.0f;
static const double CHECKSUM_BOUND_FLOAT64 = 1000000.0;

struct Request {
  String op = "divide";
  String numeric_type = "float32";
  uint32_t samples = 1000;
  uint32_t passes = 1000;
  uint32_t seed = 20260912;
};

struct Pair {
  float a;
  float b;
};

static uint32_t lcg_next(uint32_t &state) {
  state = state * 1664525UL + 1013904223UL;
  return state;
}

static float unit_float(uint32_t &state) {
  return (lcg_next(state) >> 8) * (1.0f / 16777215.0f);
}

static float uniform_float(uint32_t &state, float low, float high) {
  return low + (high - low) * unit_float(state);
}

static Pair sample_pair(uint32_t seed, uint32_t sample_index) {
  uint32_t state = seed ^ (sample_index * 747796405UL + 2891336453UL);
  float a = uniform_float(state, -1000.0f, 1000.0f);
  float b = uniform_float(state, 0.01f, 1000.0f);
  if ((lcg_next(state) & 1UL) != 0) {
    b = -b;
  }
  return Pair{a, b};
}

static String extract_string(const String &line, const char *key, const String &fallback) {
  String pattern = String("\"") + key + "\"";
  int key_pos = line.indexOf(pattern);
  if (key_pos < 0) return fallback;
  int colon = line.indexOf(':', key_pos + pattern.length());
  if (colon < 0) return fallback;
  int first_quote = line.indexOf('"', colon + 1);
  if (first_quote < 0) return fallback;
  int second_quote = line.indexOf('"', first_quote + 1);
  if (second_quote < 0) return fallback;
  return line.substring(first_quote + 1, second_quote);
}

static uint32_t extract_u32(const String &line, const char *key, uint32_t fallback) {
  String pattern = String("\"") + key + "\"";
  int key_pos = line.indexOf(pattern);
  if (key_pos < 0) return fallback;
  int colon = line.indexOf(':', key_pos + pattern.length());
  if (colon < 0) return fallback;
  int start = colon + 1;
  while (start < line.length() && isspace(line[start])) {
    start++;
  }
  int end = start;
  while (end < line.length() && isdigit(line[end])) {
    end++;
  }
  if (end == start) return fallback;
  return static_cast<uint32_t>(line.substring(start, end).toInt());
}

static Request parse_request(const String &line) {
  Request request;
  request.op = extract_string(line, "op", request.op);
  request.numeric_type = extract_string(line, "numeric_type", request.numeric_type);
  request.samples = extract_u32(line, "samples", request.samples);
  request.passes = extract_u32(line, "passes", request.passes);
  request.seed = extract_u32(line, "seed", request.seed);
  return request;
}

static float apply_operation_float32(float a, float b, const String &op) {
  if (op == "add") return a + b;
  if (op == "subtract") return a - b;
  if (op == "multiply") return a * b;
  if (op == "divide") return a / b;
  return NAN;
}

static double apply_operation_float64(double a, double b, const String &op) {
  if (op == "add") return a + b;
  if (op == "subtract") return a - b;
  if (op == "multiply") return a * b;
  if (op == "divide") return a / b;
  return NAN;
}

static float bounded_accumulate_float32(float checksum, float value) {
  checksum += value;
  if (checksum > CHECKSUM_BOUND_FLOAT32) {
    checksum -= 2.0f * CHECKSUM_BOUND_FLOAT32;
  } else if (checksum < -CHECKSUM_BOUND_FLOAT32) {
    checksum += 2.0f * CHECKSUM_BOUND_FLOAT32;
  }
  return checksum;
}

static double bounded_accumulate_float64(double checksum, double value) {
  checksum += value;
  if (checksum > CHECKSUM_BOUND_FLOAT64) {
    checksum -= 2.0 * CHECKSUM_BOUND_FLOAT64;
  } else if (checksum < -CHECKSUM_BOUND_FLOAT64) {
    checksum += 2.0 * CHECKSUM_BOUND_FLOAT64;
  }
  return checksum;
}

static double run_kernel_float32(const Request &request) {
  float checksum = 0.0f;
  for (uint32_t pass = 0; pass < request.passes; ++pass) {
    for (uint32_t i = 0; i < request.samples; ++i) {
      Pair pair = sample_pair(request.seed, i);
      float a = pair.a;
      float b = pair.b;
      checksum = bounded_accumulate_float32(checksum, apply_operation_float32(a, b, request.op));
    }
  }
  g_result_sink += static_cast<double>(checksum);
  return static_cast<double>(checksum);
}

static double run_kernel_float64(const Request &request) {
  double checksum = 0.0;
  for (uint32_t pass = 0; pass < request.passes; ++pass) {
    for (uint32_t i = 0; i < request.samples; ++i) {
      Pair pair = sample_pair(request.seed, i);
      double a = static_cast<double>(pair.a);
      double b = static_cast<double>(pair.b);
      checksum = bounded_accumulate_float64(checksum, apply_operation_float64(a, b, request.op));
    }
  }
  g_result_sink += checksum;
  return checksum;
}

static double run_empty_loop(const Request &request) {
  double checksum = 0.0;
  for (uint32_t pass = 0; pass < request.passes; ++pass) {
    for (uint32_t i = 0; i < request.samples; ++i) {
      checksum = bounded_accumulate_float64(checksum, static_cast<double>(i & 1UL));
    }
  }
  g_result_sink += checksum;
  return checksum;
}

static void print_profile() {
  Serial.print("{\"backend\":\"esp32_arduino\"");
  Serial.print(",\"status\":\"ok\"");
  Serial.print(",\"op\":\"profile\"");
  Serial.print(",\"chip_model\":\"");
  Serial.print(ESP.getChipModel());
  Serial.print("\"");
  Serial.print(",\"chip_revision\":");
  Serial.print(ESP.getChipRevision());
  Serial.print(",\"chip_cores\":");
  Serial.print(ESP.getChipCores());
  Serial.print(",\"cpu_frequency_mhz\":");
  Serial.print(ESP.getCpuFreqMHz());
  Serial.print(",\"flash_size_bytes\":");
  Serial.print(ESP.getFlashChipSize());
  Serial.print(",\"free_heap_bytes\":");
  Serial.print(ESP.getFreeHeap());
  Serial.print(",\"min_free_heap_bytes\":");
  Serial.print(ESP.getMinFreeHeap());
  Serial.print(",\"double_size_bytes\":");
  Serial.print(sizeof(double));
  Serial.println("}");
}

static bool valid_request(const Request &request, String &error) {
  if (request.op == "profile" || request.op == "suite") return true;
  if (!(request.op == "empty_loop" || request.op == "add" || request.op == "subtract" ||
        request.op == "multiply" || request.op == "divide")) {
    error = "UNKNOWN_OPERATION";
    return false;
  }
  if (!(request.numeric_type == "float32" || request.numeric_type == "float64")) {
    error = "UNSUPPORTED_NUMERIC_TYPE";
    return false;
  }
  if (request.samples == 0 || request.passes == 0) {
    error = "INVALID_SAMPLE_OR_PASS_COUNT";
    return false;
  }
  if (request.samples > 100000UL || request.passes > 100000UL) {
    error = "REQUEST_TOO_LARGE";
    return false;
  }
  return true;
}

static void print_error(const Request &request, const String &error) {
  Serial.print("{\"backend\":\"esp32_arduino\"");
  Serial.print(",\"status\":\"error\"");
  Serial.print(",\"op\":\"");
  Serial.print(request.op);
  Serial.print("\"");
  Serial.print(",\"numeric_type\":\"");
  Serial.print(request.numeric_type);
  Serial.print("\"");
  Serial.print(",\"error\":\"");
  Serial.print(error);
  Serial.println("\"}");
}

static void run_and_print_benchmark(const Request &request) {
  const uint64_t measured_evaluations =
      static_cast<uint64_t>(request.samples) * static_cast<uint64_t>(request.passes);
  const uint32_t heap_before = ESP.getFreeHeap();
  const uint32_t min_heap_before = ESP.getMinFreeHeap();
  const uint32_t start_us = micros();
  double checksum = 0.0;

  if (request.op == "empty_loop") {
    checksum = run_empty_loop(request);
  } else if (request.numeric_type == "float32") {
    checksum = run_kernel_float32(request);
  } else {
    checksum = run_kernel_float64(request);
  }

  const uint32_t elapsed_us = micros() - start_us;
  const uint32_t heap_after = ESP.getFreeHeap();
  const uint32_t min_heap_after = ESP.getMinFreeHeap();
  const double runtime_per_eval_ns =
      measured_evaluations == 0 ? 0.0 : (static_cast<double>(elapsed_us) * 1000.0) /
                                           static_cast<double>(measured_evaluations);

  Serial.print("{\"backend\":\"esp32_arduino\"");
  Serial.print(",\"status\":\"ok\"");
  Serial.print(",\"op\":\"");
  Serial.print(request.op);
  Serial.print("\"");
  Serial.print(",\"numeric_type\":\"");
  Serial.print(request.numeric_type);
  Serial.print("\"");
  Serial.print(",\"samples\":");
  Serial.print(request.samples);
  Serial.print(",\"passes\":");
  Serial.print(request.passes);
  Serial.print(",\"measured_evaluations\":");
  Serial.print(static_cast<unsigned long>(measured_evaluations));
  Serial.print(",\"seed\":");
  Serial.print(request.seed);
  Serial.print(",\"total_runtime_us\":");
  Serial.print(elapsed_us);
  Serial.print(",\"runtime_per_eval_ns\":");
  Serial.print(runtime_per_eval_ns, 6);
  Serial.print(",\"result_checksum\":");
  if (isfinite(checksum)) {
    Serial.print(checksum, 12);
  } else {
    Serial.print("\"ovf\"");
  }
  Serial.print(",\"checksum_method\":\"bounded_sum_v1\"");
  Serial.print(",\"free_heap_before\":");
  Serial.print(heap_before);
  Serial.print(",\"free_heap_after\":");
  Serial.print(heap_after);
  Serial.print(",\"min_free_heap_before\":");
  Serial.print(min_heap_before);
  Serial.print(",\"min_free_heap_after\":");
  Serial.print(min_heap_after);
  Serial.print(",\"double_size_bytes\":");
  Serial.print(sizeof(double));
  Serial.println("}");
}

static void run_suite(const Request &base_request) {
  const char *ops[] = {"empty_loop", "add", "subtract", "multiply", "divide"};
  const char *types[] = {"float32", "float64"};

  Serial.print("{\"backend\":\"esp32_arduino\",\"status\":\"suite_start\"");
  Serial.print(",\"samples\":");
  Serial.print(base_request.samples);
  Serial.print(",\"passes\":");
  Serial.print(base_request.passes);
  Serial.print(",\"seed\":");
  Serial.print(base_request.seed);
  Serial.println("}");

  for (size_t type_i = 0; type_i < 2; ++type_i) {
    for (size_t op_i = 0; op_i < 5; ++op_i) {
      Request request = base_request;
      request.op = ops[op_i];
      request.numeric_type = types[type_i];
      run_and_print_benchmark(request);
      delay(10);
    }
  }

  Serial.println("{\"backend\":\"esp32_arduino\",\"status\":\"suite_end\"}");
}

static void handle_request_line(const String &line) {
  Request request = parse_request(line);
  String error;
  if (!valid_request(request, error)) {
    print_error(request, error);
    return;
  }

  if (request.op == "profile") {
    print_profile();
    return;
  }

  if (request.op == "suite") {
    run_suite(request);
    return;
  }

  run_and_print_benchmark(request);
}

void setup() {
  Serial.begin(BAUD_RATE);
  while (!Serial) {
    delay(10);
  }
  Serial.println("{\"backend\":\"esp32_arduino\",\"status\":\"ready\",\"baud\":115200}");
}

void loop() {
  while (Serial.available() > 0) {
    char ch = static_cast<char>(Serial.read());
    if (ch == '\r') {
      continue;
    }
    if (ch == '\n') {
      g_line[g_line_len] = '\0';
      if (g_line_len > 0) {
        handle_request_line(String(g_line));
      }
      g_line_len = 0;
      continue;
    }
    if (g_line_len + 1 < MAX_LINE) {
      g_line[g_line_len++] = ch;
    } else {
      g_line_len = 0;
      Serial.println("{\"backend\":\"esp32_arduino\",\"status\":\"error\",\"error\":\"LINE_TOO_LONG\"}");
    }
  }
}
