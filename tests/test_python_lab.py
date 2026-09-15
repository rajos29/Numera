import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_lab"))

import arithmetic_lab


class ArithmeticLabTests(unittest.TestCase):
    def test_shared_inputs_are_deterministic(self):
        params = dict(arithmetic_lab.DEFAULT_INPUT_PARAMETERS)
        first = arithmetic_lab.deterministic_input_set(
            "float64", 123, 10, "uniform_moderate_magnitude", params
        )
        second = arithmetic_lab.deterministic_input_set(
            "float64", 123, 10, "uniform_moderate_magnitude", params
        )
        self.assertEqual(first, second)

    def test_shared_inputs_are_operation_independent(self):
        params = dict(arithmetic_lab.DEFAULT_INPUT_PARAMETERS)
        inputs = arithmetic_lab.deterministic_input_set(
            "float64", 123, 10, "uniform_moderate_magnitude", params
        )
        add_first = arithmetic_lab.apply_operation(inputs[0][0], inputs[0][1], "add", "float64")
        subtract_first = arithmetic_lab.apply_operation(
            inputs[0][0], inputs[0][1], "subtract", "float64"
        )
        self.assertNotEqual(add_first, subtract_first)
        self.assertEqual(
            inputs,
            arithmetic_lab.deterministic_input_set(
                "float64", 123, 10, "uniform_moderate_magnitude", params
            ),
        )

    def test_relative_error_is_not_reported_near_zero_reference(self):
        absolute, relative = arithmetic_lab.calculate_reference_error(
            checksum=0.0,
            inputs=[(1.0, -1.0)],
            operation="add",
        )
        self.assertEqual(absolute, 0.0)
        self.assertIsNone(relative)


if __name__ == "__main__":
    unittest.main()
