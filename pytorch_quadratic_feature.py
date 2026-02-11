#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pytorch_quadratic_feature.py — Quadratic Feature (Polynomial Regression) with NumPy + PyTorch
(+ optional TensorFlow + overfitting demo)

GOAL
----
Show a simple but very important ML idea:

➡️ Instead of using only x as a feature, we also add x² as an extra feature.

This turns a "linear" model (a straight line) into a "quadratic" model (a parabola).

    Linear model:
        y_pred = w1 * x + b

    Quadratic model (with the extra feature x²):
        y_pred = w1 * x + w2 * x² + b

IMPORTANT
---------
Even if the curve is a parabola as a function of x,
the model is still *linear with respect to the parameters* (w1, w2, b).
That’s why we can still learn it with gradient descent / SGD.

HOW TO RUN
----------
(NumPy only)
    python3 -m pip install numpy
    python3 pytorch_quadratic_feature.py

(NumPy + PyTorch)
    python3 -m pip install numpy torch
    python3 pytorch_quadratic_feature.py --do_torch

(Optional TensorFlow)
    python3 -m pip install tensorflow numpy
    python3 pytorch_quadratic_feature.py --do_tf

(Overfitting demo: too many features)
    python3 -m pip install tensorflow numpy matplotlib
    python3 pytorch_quadratic_feature.py --do_overfit
"""

from __future__ import annotations  # allows using future-style type hints in older Python versions

import argparse  # for command-line arguments like --do_torch
from typing import Tuple  # for type hints like Tuple[np.ndarray, np.ndarray]
import numpy as np  # main numerical library (arrays, matrix ops, etc.)


# =========================================================
# 0) PRINTING HELPERS (just to display things nicely)
# =========================================================
def title(s: str) -> None:
    print("\n" + "=" * 78)      # separator line (visual)
    print(s)                   # section title
    print("=" * 78)            # separator line (visual)


def show_io(label_in: str, x_in, label_out: str, x_out) -> None:
    """Pretty print an input -> output example."""
    print(f"\n{label_in}:")     # label for the "input"
    print(x_in)                 # actual input values
    print(f"\n{label_out}:")    # label for the "output"
    print(x_out)                # actual output values


# =========================================================
# 1) DATASET: base data (x and y) + features [x, x^2]
# =========================================================
def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X : array (N,2) where each row = [x, x^2]
        y : array (N,) prices

    Base data (like in the slide):
        x = [2, 3, 4, 6, 7]
        y = [70, 110, 165, 390, 550]
    """
    # x = phone index (e.g., 2 means "Galaxy S2")
    x = np.array([2, 3, 4, 6, 7], dtype=np.float64)   # shape (N,)

    # y = phone price (target values we want to predict)
    y = np.array([70, 110, 165, 390, 550], dtype=np.float64)  # shape (N,)

    # Build features: first column is x, second column is x^2
    # Example: x=3 -> [3, 9]
    X = np.stack([x, x**2], axis=1)   # shape (N,2)

    return X, y


def explain_dataset() -> None:
    title("0) DATASET: base data and feature transformation")

    X, y = get_dataset()             # load X and y
    x = X[:, 0]                      # take the first column of X (the raw x values)

    show_io("Raw x values (phone index)", x,
            "Features X = [x, x^2] (what the model sees)", X)

    show_io("Targets y (price)", y, "Number of examples N", len(y))

    print("\nSimple interpretation:")
    print("- We only have 5 training points.")
    print("- With so few points, adding too many features can easily overfit.")
    print("- Here we add only x^2: enough to capture a simple curve (parabola).")


# =========================================================
# 2) IDEA: why add x^2?
# =========================================================
def demo_quadratic_idea() -> None:
    title("1) WHY add a quadratic feature (x^2)?")

    print("Problem:")
    print("- If the relationship between x and y is curved, a straight line is not enough.")
    print("- A straight line can underfit.\n")

    print("Solution:")
    print("- Keep linear regression, BUT enrich the input features.")
    print("- Give the model [x, x^2] instead of only [x].\n")

    print("Model:")
    print("  y_pred = w1*x + w2*x^2 + b\n")

    print("What changes:")
    print("- Instead of a line, y_pred becomes a parabola as a function of x.")
    print("- But it is still a weighted sum of features (easy to optimize).")


# =========================================================
# 3) NUMPY: proper gradient descent on MSE
# =========================================================
def numpy_train_quadratic(iters: int, lr: float, verbose_every: int) -> Tuple[np.ndarray, float]:
    """
    Train w and b by minimizing Mean Squared Error (MSE).

    Model:
        y_pred = X @ w + b
        where X = [x, x^2] and w = [w1, w2]

    MSE:
        MSE = (1/N) * sum (y_pred - y)^2

    Gradient:
        r = (y_pred - y)
        grad_w = (2/N) * X^T r
        grad_b = (2/N) * sum(r)

    Update:
        w = w - lr * grad_w
        b = b - lr * grad_b
    """
    title("2) NUMPY: training (minimize MSE with Gradient Descent)")

    X, y = get_dataset()          # load dataset
    N = len(y)                    # number of samples (here N=5)

    w = np.zeros(2, dtype=np.float64)  # initialize weights [w1, w2] to 0
    b = 0.0                            # initialize bias to 0

    for i in range(iters):             # repeat many times to converge
        y_pred = X @ w + b             # forward pass: compute predictions (vector of length N)

        r = (y_pred - y)               # residuals/errors: prediction - true value

        grad_w = (2.0 / N) * (X.T @ r) # gradient for weights (shape (2,))
        grad_b = (2.0 / N) * np.sum(r) # gradient for bias (scalar)

        w -= lr * grad_w               # update weights (move opposite the gradient)
        b -= lr * grad_b               # update bias

        if verbose_every and (i % verbose_every == 0):
            mse = np.mean((y_pred - y) ** 2)  # compute current MSE for monitoring
            print(f"iter={i:>7} | w1={w[0]:>10.4f} w2={w[1]:>10.4f} b={b:>10.4f} | mse={mse:>12.4f}")

    print("\nEnd of NumPy training:")
    print(f"- w = [w1, w2] = {w}")      # final weights
    print(f"- b = {b:.6f}")             # final bias

    return w, b


def numpy_predict(x: float, w: np.ndarray, b: float) -> float:
    """Predict y from a single x using [x, x^2]."""
    return w[0] * x + w[1] * (x ** 2) + b   # compute w1*x + w2*x^2 + b


def show_predictions(w: np.ndarray, b: float) -> None:
    title("3) PREDICTIONS: Galaxy S5 and Galaxy S1 examples")

    pred_s5 = numpy_predict(5.0, w, b)  # x=5 => features [5,25]
    pred_s1 = numpy_predict(1.0, w, b)  # x=1 => features [1,1]

    show_io("Input Galaxy S5: x=5 -> [x, x^2]", np.array([5.0, 25.0]),
            "Predicted price", f"${pred_s5:.2f}")

    show_io("Input Galaxy S1: x=1 -> [x, x^2]", np.array([1.0, 1.0]),
            "Predicted price", f"${pred_s1:.2f}")

    print("\nImportant note:")
    print("- With only 5 points, the model may not generalize perfectly.")
    print("- It captures a trend, not a guaranteed exact price.")


# =========================================================
# 4) PYTORCH: same model, autograd computes gradients
# =========================================================
def demo_pytorch_quadratic(iters: int, lr: float, verbose_every: int) -> None:
    title("4) PYTORCH: same model, automatic gradients (autograd) + SGD")

    try:
        import torch                 # core PyTorch
        import torch.optim as optim  # optimizers like SGD
    except Exception as e:
        print("PyTorch is not available.")
        print("Install with: python3 -m pip install torch")
        print("Error:", e)
        return

    X_np, y_np = get_dataset()                         # get NumPy arrays
    X = torch.tensor(X_np, dtype=torch.float32)        # convert features to a torch tensor (N,2)
    y = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float32)  # convert targets to (N,1)

    W = torch.zeros((2, 1), requires_grad=True)  # weights tensor (2,1), requires_grad=True enables autograd
    b = torch.zeros(1, requires_grad=True)       # bias scalar, also tracked by autograd

    optimizer = optim.SGD([W, b], lr=lr)         # SGD optimizer that updates W and b

    for i in range(iters):
        pred = X @ W + b                         # forward: matrix multiply + bias (N,1)

        loss = ((pred - y) ** 2).mean()          # MSE loss

        optimizer.zero_grad()                    # reset gradients from previous iteration
        loss.backward()                          # compute gradients dloss/dW and dloss/db automatically
        optimizer.step()                         # update W and b using computed gradients

        if verbose_every and (i % verbose_every == 0):
            with torch.no_grad():                # disable gradient tracking for printing
                w_view = W.view(-1).cpu().numpy()
                print(f"iter={i:>7} | W={w_view} b={b.item():.4f} | loss={loss.item():.4f}")

    with torch.no_grad():                         # safe extraction of learned parameters
        w1, w2 = float(W[0, 0].item()), float(W[1, 0].item())
        b0 = float(b.item())

    title("4b) PYTORCH: predictions (same inputs)")
    s5 = w1 * 5.0 + w2 * 25.0 + b0               # prediction for x=5
    s1 = w1 * 1.0 + w2 * 1.0 + b0                # prediction for x=1
    show_io("Input Galaxy S5: [5,25]", np.array([5.0, 25.0]), "Predicted price", f"${s5:.2f}")
    show_io("Input Galaxy S1: [1,1]", np.array([1.0, 1.0]), "Predicted price", f"${s1:.2f}")


# =========================================================
# 5) TENSORFLOW (optional): TF1 style via tf.compat.v1
# =========================================================
def demo_tensorflow_quadratic(iters: int, lr: float, verbose_every: int) -> None:
    title("5) TENSORFLOW (optional): TF1-style version (tf.compat.v1)")

    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow is not available.")
        print("Install with: python3 -m pip install tensorflow")
        print("Error:", e)
        return

    tf.compat.v1.disable_eager_execution()            # use TF1 graph mode (placeholders, sessions)

    X_np, y_np = get_dataset()                        # get data from our helper
    data_x = X_np.astype(np.float32)                  # ensure float32 for TF
    data_y = y_np.reshape(-1, 1).astype(np.float32)   # ensure shape (N,1)

    features = 2
    x = tf.compat.v1.placeholder(tf.float32, [None, features])  # input placeholder
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 1])        # target placeholder

    W = tf.Variable(tf.zeros([features, 1], dtype=tf.float32))  # weights variable
    b = tf.Variable(tf.zeros([1], dtype=tf.float32))            # bias variable

    pred = tf.matmul(x, W) + b                         # forward computation graph
    loss = tf.reduce_mean(tf.pow(pred - y_, 2))        # MSE loss node
    update = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(loss)  # training op

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())   # init W and b

        for i in range(iters):
            sess.run(update, feed_dict={x: data_x, y_: data_y}) # run one training step
            if verbose_every and (i % verbose_every == 0):
                W_val, b_val, loss_val = sess.run([W, b, loss], feed_dict={x: data_x, y_: data_y})
                print(f"iter={i:>7} | W={W_val.reshape(-1)} b={b_val.item():.4f} | loss={loss_val:.4f}")

        W_val, b_val = sess.run([W, b])                # fetch final values

    w1, w2 = float(W_val[0, 0]), float(W_val[1, 0])    # extract weights
    b0 = float(b_val[0])                                # extract bias

    title("5b) TF: predictions")
    s5 = w1 * 5.0 + w2 * 25.0 + b0
    s1 = w1 * 1.0 + w2 * 1.0 + b0
    show_io("Input Galaxy S5: [5,25]", np.array([5.0, 25.0]), "Predicted price", f"${s5:.2f}")
    show_io("Input Galaxy S1: [1,1]", np.array([1.0, 1.0]), "Predicted price", f"${s1:.2f}")


# =========================================================
# 6) OVERFITTING: too many polynomial features (degree 20)
# =========================================================
def demo_too_many_features_tf(iters: int, lr: float, features: int, seed: int, plot: bool, verbose_every: int) -> None:
    title("6) BONUS: too many features (degree 20) => overfitting risk")

    print("Idea:")
    print("- We keep only 5 points, but create 20 features (x^1 ... x^20).")
    print("- The model becomes extremely flexible and can produce a weird curve.")
    print("- It can fit the training points perfectly, but be terrible in-between (overfitting).\n")

    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow is not available.")
        print("Install with: python3 -m pip install tensorflow")
        print("Error:", e)
        return

    tf.compat.v1.disable_eager_execution()             # graph mode

    rng = np.random.default_rng(seed)                 # RNG for reproducibility
    degrees = np.arange(1, features + 1)              # [1,2,...,features]
    rng.shuffle(degrees)                              # shuffle degrees (like the slide)

    def vecto(x: float) -> np.ndarray:
        # Create polynomial features: [x^d / 7^d for d in degrees]
        # We divide by 7^d so the values don't explode too fast.
        return np.array([x ** d / (7.0 ** d) for d in degrees], dtype=np.float32)

    X_train = np.array([vecto(2), vecto(3), vecto(4), vecto(6), vecto(7)], dtype=np.float32)  # (5,features)
    _, y_np = get_dataset()
    y_train = y_np.reshape(-1, 1).astype(np.float32)  # (5,1)

    x_ph = tf.compat.v1.placeholder(tf.float32, [None, features])  # input placeholder
    y_ph = tf.compat.v1.placeholder(tf.float32, [None, 1])         # target placeholder

    W = tf.Variable(tf.zeros([features, 1], dtype=tf.float32))     # weights for all features
    b = tf.Variable(tf.zeros([1], dtype=tf.float32))               # bias

    y_pred = tf.matmul(x_ph, W) + b                      # model prediction
    loss = tf.reduce_mean(tf.pow(y_pred - y_ph, 2))      # training loss
    update = tf.compat.v1.train.GradientDescentOptimizer(lr).minimize(loss)  # training op

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())      # init variables

        for i in range(iters):
            sess.run(update, feed_dict={x_ph: X_train, y_ph: y_train})  # training step
            if verbose_every and (i % verbose_every == 0):
                loss_val = sess.run(loss, feed_dict={x_ph: X_train, y_ph: y_train})
                print(f"iter={i:>7} | train_loss={loss_val:.6f}")

        if not plot:
            print("\nPlot disabled (--no_plot). Done.")
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("matplotlib is not available.")
            print("Install with: python3 -m pip install matplotlib")
            print("Error:", e)
            return

        x_axis = np.arange(0.0, 8.0, 0.1, dtype=np.float32)      # grid of x values for plotting
        X_grid = np.array([vecto(float(xv)) for xv in x_axis], dtype=np.float32)  # features for the grid

        W_val, b_val = sess.run([W, b])                          # get learned weights and bias
        y_grid = X_grid @ W_val + b_val                          # compute predictions on the grid

        plt.figure()
        plt.plot(x_axis, y_grid.reshape(-1))                      # predicted curve
        plt.scatter(np.array([2, 3, 4, 6, 7], dtype=np.float32), y_np, marker="x")  # training points
        plt.title("Too many features (degree 20): potentially overfit curve")
        plt.xlabel("x")
        plt.ylabel("y (price)")
        plt.show()


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Quadratic feature demo (NumPy + PyTorch) + optional TF + overfitting.")

    # NumPy training settings
    p.add_argument("--iters_numpy", type=int, default=300_000)     # how many GD steps
    p.add_argument("--lr_numpy", type=float, default=1e-3)         # learning rate
    p.add_argument("--verbose_numpy", type=int, default=50_000)    # print every N steps

    # PyTorch optional settings
    p.add_argument("--do_torch", action="store_true")              # run PyTorch part if passed
    p.add_argument("--iters_torch", type=int, default=100_000)
    p.add_argument("--lr_torch", type=float, default=1e-3)
    p.add_argument("--verbose_torch", type=int, default=10_000)

    # TensorFlow optional settings
    p.add_argument("--do_tf", action="store_true")                 # run TF part if passed
    p.add_argument("--iters_tf", type=int, default=100_000)
    p.add_argument("--lr_tf", type=float, default=1e-3)
    p.add_argument("--verbose_tf", type=int, default=10_000)

    # Overfitting optional settings
    p.add_argument("--do_overfit", action="store_true")            # run overfit demo if passed
    p.add_argument("--overfit_features", type=int, default=20)     # number of polynomial features
    p.add_argument("--overfit_iters", type=int, default=200_000)
    p.add_argument("--overfit_lr", type=float, default=0.1)
    p.add_argument("--overfit_seed", type=int, default=0)          # RNG seed for reproducibility
    p.add_argument("--overfit_verbose", type=int, default=50_000)
    p.add_argument("--no_plot", action="store_true")               # disable plot if passed

    args = p.parse_args()                                          # parse CLI arguments

    explain_dataset()                                              # print dataset explanation
    demo_quadratic_idea()                                          # print intuition explanation

    w, b = numpy_train_quadratic(                                  # train NumPy model
        iters=args.iters_numpy,
        lr=args.lr_numpy,
        verbose_every=args.verbose_numpy,
    )
    show_predictions(w, b)                                         # show predictions

    if args.do_torch:                                              # run PyTorch part only if asked
        demo_pytorch_quadratic(
            iters=args.iters_torch,
            lr=args.lr_torch,
            verbose_every=args.verbose_torch,
        )

    if args.do_tf:                                                 # run TF part only if asked
        demo_tensorflow_quadratic(
            iters=args.iters_tf,
            lr=args.lr_tf,
            verbose_every=args.verbose_tf,
        )

    if args.do_overfit:                                            # run overfitting demo only if asked
        demo_too_many_features_tf(
            iters=args.overfit_iters,
            lr=args.overfit_lr,
            features=args.overfit_features,
            seed=args.overfit_seed,
            plot=(not args.no_plot),
            verbose_every=args.overfit_verbose,
        )

    title("Done ✅")
    print("Script finished. You can tune iterations / lr / verbose if needed.")


if __name__ == "__main__":
    main()  # entry point: runs main() when you execute the file