#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

volatile double g_result_sink = 0.0;

struct Args {
    std::string operation;
    std::string numeric_type;
    std::string input_path;
    std::string experiment_id;
    std::string input_class;
    std::string input_set_id;
    int trial_id = 0;
    int seed = 0;
    std::size_t passes = 1;
};

struct Pair {
    double a;
    double b;
};

void print_usage() {
    std::cerr
        << "Usage: arithmetic_benchmark --operation OP --numeric-type TYPE "
        << "--input PATH --experiment-id ID --input-class CLASS "
        << "--input-set-id ID --trial-id N --seed N\n";
}

bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (i + 1 >= argc) {
            return false;
        }
        std::string value = argv[++i];
        if (key == "--operation") {
            args.operation = value;
        } else if (key == "--numeric-type") {
            args.numeric_type = value;
        } else if (key == "--input") {
            args.input_path = value;
        } else if (key == "--experiment-id") {
            args.experiment_id = value;
        } else if (key == "--input-class") {
            args.input_class = value;
        } else if (key == "--input-set-id") {
            args.input_set_id = value;
        } else if (key == "--trial-id") {
            args.trial_id = std::stoi(value);
        } else if (key == "--seed") {
            args.seed = std::stoi(value);
        } else if (key == "--passes") {
            args.passes = static_cast<std::size_t>(std::stoull(value));
        } else {
            return false;
        }
    }
    return !args.operation.empty() && !args.numeric_type.empty() && !args.input_path.empty();
}

std::vector<Pair> read_inputs(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open input file");
    }

    std::vector<Pair> pairs;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line == "a,b") {
            continue;
        }
        std::stringstream ss(line);
        std::string a_text;
        std::string b_text;
        if (!std::getline(ss, a_text, ',') || !std::getline(ss, b_text, ',')) {
            throw std::runtime_error("malformed input row");
        }
        pairs.push_back({std::stod(a_text), std::stod(b_text)});
    }
    return pairs;
}

template <typename T>
T apply_operation(T a, T b, const std::string& operation) {
    if (operation == "add") {
        return static_cast<T>(a + b);
    }
    if (operation == "subtract") {
        return static_cast<T>(a - b);
    }
    if (operation == "multiply") {
        return static_cast<T>(a * b);
    }
    if (operation == "divide") {
        return static_cast<T>(a / b);
    }
    return std::numeric_limits<T>::quiet_NaN();
}

template <typename T>
double run_kernel(const std::vector<Pair>& inputs, const std::string& operation, std::size_t passes) {
    T checksum = static_cast<T>(0);
    for (std::size_t pass = 0; pass < passes; ++pass) {
        for (const auto& pair : inputs) {
            T a = static_cast<T>(pair.a);
            T b = static_cast<T>(pair.b);
            checksum = static_cast<T>(checksum + apply_operation(a, b, operation));
        }
    }
    g_result_sink += static_cast<double>(checksum);
    return static_cast<double>(checksum);
}

double run_empty_loop(std::size_t samples, std::size_t passes) {
    double checksum = 0.0;
    for (std::size_t pass = 0; pass < passes; ++pass) {
        for (std::size_t i = 0; i < samples; ++i) {
            checksum += static_cast<double>(i & 1U);
        }
    }
    g_result_sink += checksum;
    return checksum;
}

std::string escape_json(const std::string& value) {
    std::string out;
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    return out;
}

void print_json_string_field(const char* key, const std::string& value, bool comma = true) {
    std::cout << "\"" << key << "\":\"" << escape_json(value) << "\"";
    if (comma) {
        std::cout << ",";
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage();
        return 2;
    }

    try {
        const auto inputs = read_inputs(args.input_path);
        const auto start = std::chrono::steady_clock::now();
        double checksum = 0.0;
        if (args.operation == "empty_loop") {
            checksum = run_empty_loop(inputs.size(), args.passes);
        } else if (args.numeric_type == "float32") {
            checksum = run_kernel<float>(inputs, args.operation, args.passes);
        } else if (args.numeric_type == "float64") {
            checksum = run_kernel<double>(inputs, args.operation, args.passes);
        } else {
            throw std::runtime_error("unsupported numeric type");
        }
        const auto stop = std::chrono::steady_clock::now();
        const auto total_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        const auto measured_evaluations = inputs.size() * args.passes;
        const double per_operation =
            measured_evaluations == 0
                ? 0.0
                : static_cast<double>(total_ns) / static_cast<double>(measured_evaluations);

        std::cout << std::setprecision(17) << "{";
        print_json_string_field("backend", "native_cpp");
        print_json_string_field("operation", args.operation);
        print_json_string_field("numeric_type", args.numeric_type);
        print_json_string_field("implementation", "cpp_builtin_operator");
        print_json_string_field("experiment_id", args.experiment_id);
        print_json_string_field("input_class", args.input_class);
        print_json_string_field("input_set_id", args.input_set_id);
        std::cout << "\"trial_id\":" << args.trial_id << ",";
        std::cout << "\"seed\":" << args.seed << ",";
        std::cout << "\"samples_per_trial\":" << inputs.size() << ",";
        std::cout << "\"passes\":" << args.passes << ",";
        std::cout << "\"measured_evaluations\":" << measured_evaluations << ",";
        std::cout << "\"total_runtime_ns\":" << total_ns << ",";
        std::cout << "\"runtime_per_operation_ns\":" << per_operation << ",";
        std::cout << "\"result_checksum\":" << checksum << ",";
        print_json_string_field("reference_method", "python_decimal_postprocess");
        print_json_string_field("status", "ok", false);
        std::cout << "}\n";
    } catch (const std::exception& exc) {
        std::cout << "{";
        print_json_string_field("backend", "native_cpp");
        print_json_string_field("operation", args.operation);
        print_json_string_field("numeric_type", args.numeric_type);
        print_json_string_field("status", "error");
        print_json_string_field("error", exc.what(), false);
        std::cout << "}\n";
        return 1;
    }

    return 0;
}
