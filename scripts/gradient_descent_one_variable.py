import numpy as np


def gradient_descent_1d(func=None, grad_func=None, start_value=2.0, learning_rate=0.1, iterations=4):
    """Run gradient descent for one variable and return the full update history.

    Step-by-step:
    1. If `func` or `grad_func` are not provided, use sensible defaults.
    2. Initialize `current_value` from `start_value` and prepare an empty `history` list.
    3. For each iteration:
       a. Evaluate gradient at the current value.
       b. Update `current_value = current_value - learning_rate * gradient`.
       c. Recompute gradient and function value at the new point.
       d. Append the iteration record (iteration number, x, f_x, gradient) to `history`.
    4. Return the final x, the history list, and the function/gradient used.
    """
    if func is None:
        func = lambda x: np.cos(x ** 2) ** 2
    if grad_func is None:
        grad_func = lambda x: -2.0 * x * np.sin(2.0 * x ** 2)

    current_value = float(start_value)
    history = []
    print(f"Initial x: {current_value:.5f}, f(x): {func(current_value):.5f}, gradient: {grad_func(current_value):.5f}")
    print("-" * 30)
    for iteration in range(1, iterations + 1):
        gradient_before_update = float(grad_func(current_value))
        current_value = float(current_value - learning_rate * gradient_before_update)
        gradient_after_update = float(grad_func(current_value))
        function_value = float(func(current_value))
        history.append(
            {
                "iteration": iteration,
                "x": current_value,
                "f_x": function_value,
                "gradient": gradient_after_update,
            }
        )
    return current_value, history, func, grad_func


if __name__ == "__main__":
    final_x, history, func, grad_func = gradient_descent_1d()
    print("Function: f(x) = cos(x^2)^2")
    print("Derivative: df/dx = -2*x*sin(2*x^2)")
    print("-" * 30)
    for step in history:
        print(f"Iteration {step['iteration']}:")
        print(f"  x        = {step['x']:.5f}")
        print(f"  f(x)     = {step['f_x']:.5f}")
        print(f"  gradient = {step['gradient']:.5f}")
        print("-" * 30)
    print("Final x:", final_x)
