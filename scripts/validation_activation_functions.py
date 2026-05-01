"""
Validation script for sigmoid and ReLU activation functions.
Checks both example datasets and generated datasets.
"""

import sys
import numpy as np

from activation_functions import relu, sigmoid


def validate_example_dataset():
	print("=" * 70)
	print("VALIDATION 1: Example Dataset")
	print("=" * 70)

	arr = np.array([-2, -1, 0, 1, 2], dtype=float)
	expected_relu = np.array([0, 0, 0, 1, 2], dtype=float)
	expected_sigmoid = 1 / (1 + np.exp(-arr))

	actual_relu = relu(arr)
	actual_sigmoid = sigmoid(arr)

	print("Input:", arr)
	print("ReLU:", actual_relu)
	print("Sigmoid:", np.round(actual_sigmoid, 6))

	assert np.allclose(actual_relu, expected_relu), "ReLU output mismatch on example dataset"
	assert np.allclose(actual_sigmoid, expected_sigmoid), "Sigmoid output mismatch on example dataset"
	assert np.array_equal(actual_relu, np.maximum(0, arr)), "ReLU should equal max(0, x)"

	print("✓ Example dataset matches the mathematical definitions.")


def validate_generated_dataset():
	print("\n" + "=" * 70)
	print("VALIDATION 2: Generated Dataset")
	print("=" * 70)

	arr = np.linspace(-8, 8, 9)
	relu_result = relu(arr)
	sigmoid_result = sigmoid(arr)

	print("Input:", arr)
	print("ReLU:", relu_result)
	print("Sigmoid:", np.round(sigmoid_result, 6))

	# Mathematical properties
	assert np.all(relu_result >= 0), "ReLU should never be negative"
	assert np.allclose(relu_result, np.maximum(0, arr)), "ReLU should match max(0, x)"
	assert np.all(sigmoid_result > 0) and np.all(sigmoid_result < 1), "Sigmoid should stay in (0, 1)"
	assert np.all(np.diff(sigmoid_result) > 0), "Sigmoid should be strictly increasing"

	print("✓ Generated dataset satisfies ReLU and sigmoid properties.")


def main():
	try:
		validate_example_dataset()
		validate_generated_dataset()
		print("\n" + "=" * 70)
		print("ALL VALIDATIONS PASSED ✓")
		print("=" * 70)
	except AssertionError as error:
		print(f"\n✗ VALIDATION FAILED: {error}")
		sys.exit(1)
	except Exception as error:
		print(f"\n✗ UNEXPECTED ERROR: {error}")
		sys.exit(1)


if __name__ == "__main__":
	main()
