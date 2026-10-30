"""
Linear Regression Optimization: Batch vs. Stochastic vs. Mini-Batch Gradient Descent
===================================================================================
This educational module implements Linear Regression from scratch using NumPy and compares
the three fundamental variants of Gradient Descent optimization:
1. Batch Gradient Descent (BGD)
2. Stochastic Gradient Descent (SGD)
3. Mini-Batch Gradient Descent (MBGD)

Mathematical Formulation:
-------------------------
1. Linear Hypothesis:
       y_hat = w * x + b
   where:
       - w: weight (slope)
       - b: bias (y-intercept)
       - x: input feature

2. Cost / Loss Function (Mean Squared Error - MSE):
       MSE(w, b) = (1 / N) * SUM_{i=1...N} (y_hat_i - y_i)^2
                 = (1 / N) * SUM_{i=1...N} (w * x_i + b - y_i)^2

3. Partial Derivatives (Analytical Gradients via Chain Rule):
       error_i = y_hat_i - y_i
       ∂MSE / ∂w = (2 / N) * SUM_{i=1...N} (error_i * x_i)  -->  2 * mean(error * X)
       ∂MSE / ∂b = (2 / N) * SUM_{i=1...N} (error_i)        -->  2 * mean(error)

4. Parameter Update Rules:
       w_new = w_old - learning_rate * (∂MSE / ∂w)
       b_new = b_old - learning_rate * (∂MSE / ∂b)

Variant Comparison at a Glance:
-------------------------------
| Method         | Samples per Update | Updates / Epoch | Gradient Variance | Hardware Efficiency |
|----------------|--------------------|-----------------|-------------------|---------------------|
| Batch GD       | All N samples      | 1               | None (Exact)      | High (Vectorized)   |
| Stochastic GD  | 1 sample           | N               | High (Noisy)      | Low (Scalar loops)  |
| Mini-Batch GD  | Batch size B       | N / B           | Moderate          | High (SIMD / GPU)   |
"""

import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Set directory paths for saving analysis artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# =============================================================================
# 1. SYNTHETIC DATASET GENERATION
# =============================================================================
def generate_dataset(
    n_samples: int = 100,
    slope: float = 2.0,
    intercept: float = 1.0,
    noise_std: float = 1.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 1D data with known linear ground truth and Gaussian noise:
        y = slope * X + intercept + N(0, noise_std^2)

    Parameters:
        n_samples (int): Number of data points to generate.
        slope (float): True ground truth weight (w_true = 2.0).
        intercept (float): True ground truth bias (b_true = 1.0).
        noise_std (float): Standard deviation of added Gaussian noise.
        random_seed (int): Seed for reproducible random numbers.

    Returns:
        X (np.ndarray): 1D array of input feature values.
        y (np.ndarray): 1D array of noisy target values.
    """
    np.random.seed(random_seed)
    # Generate evenly spaced values between 0 and 10
    X = np.linspace(0, 10, n_samples)
    # Target values with zero-mean Gaussian noise
    noise = np.random.normal(0, noise_std, n_samples)
    y = slope * X + intercept + noise
    return X, y


# =============================================================================
# 2. LOSS AND GRADIENT COMPUTATION
# =============================================================================
def calculate_loss(w: float, b: float, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) loss for given parameters (w, b).

    MSE measures the average squared difference between predictions and targets:
        MSE = (1 / N) * sum((w * X + b - y)^2)

    Parameters:
        w (float): Current model weight (slope).
        b (float): Current model bias (y-intercept).
        X (np.ndarray): Input feature values.
        y (np.ndarray): True target values.

    Returns:
        float: Computed Mean Squared Error.
    """
    predictions = w * X + b
    return float(np.mean((predictions - y) ** 2))


def calculate_gradients(
    w: float, b: float, X: np.ndarray, y: np.ndarray
) -> Tuple[float, float]:
    """
    Compute analytical gradients of MSE loss with respect to weight (w) and bias (b).

    Derivation using Chain Rule:
        Loss L = (1 / N) * sum( (w * X_i + b - y_i)^2 )
        Let error_i = w * X_i + b - y_i

        dL / dw = (1 / N) * sum( 2 * error_i * (d(w*X_i + b - y_i) / dw) )
                = (2 / N) * sum( error_i * X_i )
                = 2 * mean(error * X)

        dL / db = (1 / N) * sum( 2 * error_i * (d(w*X_i + b - y_i) / db) )
                = (2 / N) * sum( error_i * 1 )
                = 2 * mean(error)

    Parameters:
        w (float): Current weight.
        b (float): Current bias.
        X (np.ndarray): Input features (single sample or batch array).
        y (np.ndarray): Ground truth values (matching dimension of X).

    Returns:
        dw (float): Partial derivative dL / dw.
        db (float): Partial derivative dL / db.
    """
    predictions = w * X + b
    error = predictions - y

    dw = float(2 * np.mean(error * X))
    db = float(2 * np.mean(error))

    return dw, db


# =============================================================================
# 3. GRADIENT DESCENT OPTIMIZATION VARIANTS
# =============================================================================
def batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 200,
    initial_w: float = None,
    initial_b: float = None,
) -> Tuple[float, float, List[float], float]:
    """
    Batch Gradient Descent (BGD).

    Characteristics:
    - Calculates gradients across the ENTIRE training dataset in every step.
    - Updates parameters exactly ONCE per epoch.
    - Gradient direction is mathematically precise and smooth (zero mini-batch variance).
    - Can become computationally expensive for very large datasets that do not fit into RAM.

    Parameters:
        X (np.ndarray): Feature array.
        y (np.ndarray): Target array.
        learning_rate (float): Step size multiplier for parameter updates.
        epochs (int): Number of full passes over the dataset.
        initial_w (float, optional): Initial weight. If None, initialized randomly.
        initial_b (float, optional): Initial bias. If None, initialized randomly.

    Returns:
        w (float): Optimized weight.
        b (float): Optimized bias.
        losses (List[float]): MSE loss history recorded at the end of each epoch.
        elapsed_time (float): Training duration in seconds.
    """
    start_time = time.perf_counter()
    w = np.random.randn() if initial_w is None else initial_w
    b = np.random.randn() if initial_b is None else initial_b
    losses = []

    for _ in range(epochs):
        # Gradients computed on all N data points at once
        dw, db = calculate_gradients(w, b, X, y)
        w -= learning_rate * dw
        b -= learning_rate * db
        # Track full-dataset loss at the end of each epoch
        losses.append(calculate_loss(w, b, X, y))

    elapsed_time = time.perf_counter() - start_time
    return w, b, losses, elapsed_time


def stochastic_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.001,
    epochs: int = 200,
    initial_w: float = None,
    initial_b: float = None,
) -> Tuple[float, float, List[float], float]:
    """
    Stochastic Gradient Descent (SGD).

    Characteristics:
    - Updates parameters after evaluating EACH INDIVIDUAL sample (batch size = 1).
    - Performs N updates per epoch (e.g., 100 updates per epoch for 100 samples).
    - Gradient steps have high variance and fluctuate noisily.
    - Oscillation can help escape shallow local minima in complex loss surfaces,
      but requires a smaller learning rate (e.g., 0.001) to avoid overshooting.
    - Shuffling the data every epoch is crucial to avoid cyclic patterns.

    Parameters:
        X (np.ndarray): Feature array.
        y (np.ndarray): Target array.
        learning_rate (float): Step size (typically smaller than BGD to handle noise).
        epochs (int): Number of full passes over all samples.
        initial_w (float, optional): Initial weight.
        initial_b (float, optional): Initial bias.

    Returns:
        w (float): Optimized weight.
        b (float): Optimized bias.
        losses (List[float]): MSE loss history recorded at the end of each epoch.
        elapsed_time (float): Training duration in seconds.
    """
    start_time = time.perf_counter()
    w = np.random.randn() if initial_w is None else initial_w
    b = np.random.randn() if initial_b is None else initial_b
    losses = []

    for _ in range(epochs):
        # Randomly shuffle sample indices each epoch to prevent cyclic oscillations
        permutation = np.random.permutation(len(X))
        for i in permutation:
            # Pass a single sample as a 1D slice of size 1: X[i:i+1], y[i:i+1]
            dw, db = calculate_gradients(w, b, X[i : i + 1], y[i : i + 1])
            w -= learning_rate * dw
            b -= learning_rate * db

        # Record overall dataset loss once per epoch
        losses.append(calculate_loss(w, b, X, y))

    elapsed_time = time.perf_counter() - start_time
    return w, b, losses, elapsed_time


def mini_batch_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.005,
    epochs: int = 200,
    batch_size: int = 10,
    initial_w: float = None,
    initial_b: float = None,
) -> Tuple[float, float, List[float], float]:
    """
    Mini-Batch Gradient Descent (MBGD).

    Characteristics:
    - Splits the dataset into small batches (e.g., 10 to 64 samples).
    - Strikes the optimal balance between BGD (stability) and SGD (fast iterations).
    - Leverages vectorized linear algebra libraries (BLAS/LAPACK, CUDA) efficiently.
    - Standard optimization strategy used across deep learning frameworks (PyTorch, TensorFlow).

    Parameters:
        X (np.ndarray): Feature array.
        y (np.ndarray): Target array.
        learning_rate (float): Step size multiplier.
        epochs (int): Number of passes through the complete dataset.
        batch_size (int): Number of samples per update step.
        initial_w (float, optional): Initial weight.
        initial_b (float, optional): Initial bias.

    Returns:
        w (float): Optimized weight.
        b (float): Optimized bias.
        losses (List[float]): MSE loss history recorded at the end of each epoch.
        elapsed_time (float): Training duration in seconds.
    """
    start_time = time.perf_counter()
    w = np.random.randn() if initial_w is None else initial_w
    b = np.random.randn() if initial_b is None else initial_b
    losses = []

    for _ in range(epochs):
        # Shuffle entire dataset at the start of each epoch
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Iterate over dataset in chunks of 'batch_size'
        for start in range(0, len(X), batch_size):
            xb = X_shuffled[start : start + batch_size]
            yb = y_shuffled[start : start + batch_size]

            dw, db = calculate_gradients(w, b, xb, yb)
            w -= learning_rate * dw
            b -= learning_rate * db

        # Record overall dataset loss at end of epoch
        losses.append(calculate_loss(w, b, X, y))

    elapsed_time = time.perf_counter() - start_time
    return w, b, losses, elapsed_time


# =============================================================================
# 4. BENCHMARK AND RESULTS SUMMARY
# =============================================================================
def print_benchmark_table(
    results: Dict[str, Tuple[float, float, List[float], float]],
    true_w: float = 2.0,
    true_b: float = 1.0,
) -> None:
    """
    Print an aligned comparison table of final parameter estimates, errors, and timing.

    Parameters:
        results (dict): Dictionary mapping method name to (w, b, losses, time).
        true_w (float): Ground truth slope used in data generation.
        true_b (float): Ground truth intercept used in data generation.
    """
    header = (
        f"{'Algorithm':<16} | {'Weight (w)':<10} | {'Bias (b)':<10} | "
        f"{'w Error':<9} | {'b Error':<9} | {'Final Loss':<10} | {'Time (ms)':<9}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print("                    GRADIENT DESCENT BENCHMARK SUMMARY")
    print(f"       Ground Truth Target: y = {true_w:.1f} * x + {true_b:.1f}")
    print("=" * len(header))
    print(header)
    print(separator)

    for name, (w, b, losses, elapsed) in results.items():
        w_err = abs(w - true_w)
        b_err = abs(b - true_b)
        final_loss = losses[-1]
        elapsed_ms = elapsed * 1000.0

        print(
            f"{name:<16} | {w:10.4f} | {b:10.4f} | "
            f"{w_err:9.4f} | {b_err:9.4f} | {final_loss:10.4f} | {elapsed_ms:9.2f}"
        )
    print("=" * len(header) + "\n")


# =============================================================================
# 5. VISUALIZATION AND ANALYSIS
# =============================================================================
def plot_results(
    X: np.ndarray,
    y: np.ndarray,
    results: Dict[str, Tuple[float, float, List[float], float]],
    learning_rates: List[float] = None,
    save_fig: bool = True,
) -> None:
    """
    Plot comprehensive visual diagnostics:
    1. Loss convergence curve across epochs (BGD vs SGD vs MBGD).
    2. Regression fit lines against the data distribution.
    3. Learning rate sensitivity study on Batch Gradient Descent.

    Parameters:
        X (np.ndarray): Feature points.
        y (np.ndarray): Target values.
        results (dict): Trained model results for each method.
        learning_rates (list): List of learning rates for sensitivity testing.
        save_fig (bool): Whether to save the generated plots to the results folder.
    """
    if learning_rates is None:
        learning_rates = [0.001, 0.005, 0.01]

    # Create a 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    color_palette = {
        "Batch GD": "#1f77b4",
        "Stochastic GD": "#ff7f0e",
        "Mini-Batch GD": "#2ca02c",
    }

    # ---------------------------------------------------------
    # Subplot 1: Loss versus Epochs (Convergence Comparison)
    # ---------------------------------------------------------
    ax1 = axes[0]
    for name, (_, _, losses, _) in results.items():
        ax1.plot(
            losses,
            label=name,
            color=color_palette.get(name, None),
            linewidth=2,
            alpha=0.9,
        )

    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight="bold")
    ax1.set_title("1. Loss Convergence Curves", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # ---------------------------------------------------------
    # Subplot 2: Fitted Lines vs. Data Points
    # ---------------------------------------------------------
    ax2 = axes[1]
    # Scatter true data points
    ax2.scatter(
        X,
        y,
        color="#7f7f7f",
        alpha=0.5,
        s=25,
        edgecolor="none",
        label="Dataset (y = 2x + 1 + noise)",
    )
    # Ground Truth Reference Line
    ax2.plot(
        X,
        2.0 * X + 1.0,
        "k--",
        linewidth=1.8,
        label="Ground Truth (2x + 1)",
        alpha=0.8,
    )

    # Fitted model lines
    for name, (w, b, _, _) in results.items():
        ax2.plot(
            X,
            w * X + b,
            label=f"{name} (w={w:.2f}, b={b:.2f})",
            color=color_palette.get(name, None),
            linewidth=1.8,
        )

    ax2.set_xlabel("Feature (X)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Target (y)", fontsize=11, fontweight="bold")
    ax2.set_title("2. Linear Regression Model Fits", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # ---------------------------------------------------------
    # Subplot 3: Effect of Learning Rates on Batch GD
    # ---------------------------------------------------------
    ax3 = axes[2]
    # Use consistent initial weights for fair learning rate comparison
    init_w, init_b = 0.0, 0.0
    lr_colors = ["#9467bd", "#8c564b", "#d62728"]

    for lr, col in zip(learning_rates, lr_colors):
        _, _, losses, _ = batch_gradient_descent(
            X, y, learning_rate=lr, epochs=200, initial_w=init_w, initial_b=init_b
        )
        ax3.plot(
            losses,
            label=f"lr = {lr}",
            color=col,
            linewidth=2,
            alpha=0.85,
        )

    ax3.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mean Squared Error (MSE)", fontsize=11, fontweight="bold")
    ax3.set_title("3. Effect of Learning Rates (Batch GD)", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_fig:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_path = os.path.join(RESULTS_DIR, "gradient_descent_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Visualization] Figure saved to: {save_path}")

    # Display plot window when running in an interactive graphical environment
    if plt.get_backend().lower() not in ("agg", "cairo", "pdf", "ps", "svg"):
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# 6. MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    """
    Main driver routine:
    1. Generates the synthetic linear dataset.
    2. Runs Batch, Stochastic, and Mini-Batch Gradient Descent from identical initial weights.
    3. Prints quantitative benchmark metrics to console.
    4. Generates and displays visual comparisons.
    """
    print("[1/4] Generating synthetic dataset (100 samples, true slope=2.0, intercept=1.0)...")
    X, y = generate_dataset(n_samples=100, slope=2.0, intercept=1.0, noise_std=1.0, random_seed=42)

    # Initialize shared starting weights so all 3 methods begin at the exact same coordinates
    np.random.seed(123)
    shared_init_w = float(np.random.randn())
    shared_init_b = float(np.random.randn())
    print(f"      Initial Weights: w_0 = {shared_init_w:.4f}, b_0 = {shared_init_b:.4f}")

    print("[2/4] Training optimization algorithms (200 epochs each)...")
    # Note on learning rates:
    # - Batch GD evaluates all 100 samples per step -> smooth gradient, higher lr=0.01 is stable.
    # - Stochastic GD updates weights 100 times per epoch -> noisy gradients, requires lower lr=0.001.
    # - Mini-Batch GD updates weights 10 times per epoch (batch_size=10) -> balanced lr=0.005.
    results = {
        "Batch GD": batch_gradient_descent(
            X, y, learning_rate=0.01, epochs=200, initial_w=shared_init_w, initial_b=shared_init_b
        ),
        "Stochastic GD": stochastic_gradient_descent(
            X, y, learning_rate=0.001, epochs=200, initial_w=shared_init_w, initial_b=shared_init_b
        ),
        "Mini-Batch GD": mini_batch_gradient_descent(
            X, y, learning_rate=0.005, epochs=200, batch_size=10, initial_w=shared_init_w, initial_b=shared_init_b
        ),
    }

    print("[3/4] Compiling benchmark results...")
    print_benchmark_table(results, true_w=2.0, true_b=1.0)

    print("[4/4] Rendering analysis plots (Convergence, Fitted Models, and Learning Rate Study)...")
    plot_results(X, y, results, learning_rates=[0.001, 0.005, 0.01], save_fig=True)


if __name__ == "__main__":
    main()
