import numpy as np


def gradient_descent_2d(func=None, grad_func=None, start_x=3.0, start_y=2.0, learning_rate=0.2, iterations=5):
    """Run gradient descent for two variables and return the full update history.

    Step-by-step:
    1. Use default objective/gradient if none provided.
    2. Initialize x and y from `start_x` and `start_y`.
    3. For each iteration evaluate gradient components (grad_x, grad_y).
    4. Update each coordinate: x <- x - lr * grad_x, y <- y - lr * grad_y.
    5. Record the new coordinates, gradient components and function value into `history`.
    6. Return final coordinates, history, and the provided objective/gradient.
    """
    if func is None:
        func = lambda x, y: x ** 2
    if grad_func is None:
        grad_func = lambda x, y: (2.0 * x, 0.0)

    x_value = float(start_x)
    y_value = float(start_y)
    history = []
    print(f"Initial position: (x={x_value:.4f}, y={y_value:.4f}), f(x,y): {func(x_value, y_value):.4f}, gradient: {grad_func(x_value, y_value)}")
    print("-" * 30)
    for iteration in range(1, iterations + 1):
        grad_x, grad_y = grad_func(x_value, y_value)
        x_value = float(x_value - learning_rate * grad_x)
        y_value = float(y_value - learning_rate * grad_y)
        current_f = float(func(x_value, y_value))
        history.append(
            {
                "iteration": iteration,
                "x": x_value,
                "y": y_value,
                "gradient_x": float(grad_x),
                "gradient_y": float(grad_y),
                "f_xy": current_f,
            }
        )

    return x_value, y_value, history, func, grad_func


if __name__ == "__main__":
    x_value, y_value, history, func, grad_func = gradient_descent_2d()
    print("Function: f(x,y) = x^2")
    print("df/dx = 2*x")
    print("df/dy = 0")
    print("-" * 30)
    for step in history:
        print(f"Iteration {step['iteration']}:")
        print(f"  Gradient: ({step['gradient_x']:.4f}, {step['gradient_y']:.4f})")
        print(f"  New Position: (x={step['x']:.4f}, y={step['y']:.4f})")
        print(f"  Function Value: {step['f_xy']:.4f}")
        print("-" * 15)
    print("Final position:", (x_value, y_value))
