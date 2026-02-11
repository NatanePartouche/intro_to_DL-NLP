"""
pytorch_intro.py — Introduction to PyTorch (Tensors, Autograd, Training, Checkpoints)

This script walks through a classic PyTorch mini-ML pipeline step-by-step and explains:
- What each step is
- Why we do it
- A clear example of input -> output

Run:
    python3 -m pip install torch
    python3 pytorch_intro.py

Note:
- CUDA works only with NVIDIA GPUs and a CUDA-enabled PyTorch build.
- On Mac, you typically use CPU or MPS (Apple Silicon) instead of CUDA.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------
# Pretty printing helpers (same spirit as the NLTK file)
# ---------------------------------------------------------
def title(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


def show_io(label_in: str, x_in, label_out: str, x_out) -> None:
    """Small helper: show input -> output in a consistent format."""
    print(f"\n{label_in}:")
    print(x_in)
    print(f"\n{label_out}:")
    print(x_out)


# =========================================================
# 1) TENSORS
# =========================================================
def demo_tensors() -> None:
    """
    Tensors are PyTorch’s core data structure.

    What is a tensor?
    - A tensor is a multi-dimensional numeric container.
    - It generalizes:
        scalar (0D), vector (1D), matrix (2D), and higher dimensions (3D+).
    - A tensor has 3 key attributes:
        1) shape  -> dimensions (ex: [3,4])
        2) dtype  -> number type (int64, float32, ...)
        3) device -> where it lives (cpu, cuda)

    Why it matters:
    - Every neural network input/output is a tensor.
    - Most errors in PyTorch come from shape/dtype/device mismatches.

    This demo shows:
    - Creating a 1D tensor (vector)
    - Creating a 2D tensor (matrix)
    - Reading .shape and .dtype
    """
    title("1) TENSORS: creation, shape, dtype")

    t1 = torch.tensor([1, 2, 3, 4])
    show_io("Input (Python list)", [1, 2, 3, 4], "Output (torch.tensor)", t1)
    show_io("t1.shape", t1.shape, "t1.dtype", t1.dtype)

    t2 = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])
    show_io("Input (list of lists)", "[[1,2,3,4],[5,6,7,8],[9,10,11,12]]", "Output (tensor)", t2)
    show_io("t2.shape", t2.shape, "Interpretation", "3 rows × 4 columns")

    print("\nKey idea:")
    print("- t2.shape = (3,4) means: 3 rows, 4 columns.")
    print("- The first number is rows, the second is columns (for 2D tensors).")


# =========================================================
# 2) COMMON CREATION ERRORS
# =========================================================
def demo_tensor_creation_errors() -> None:
    """
    Common tensor creation errors.

    A) Ragged rows (non-rectangular data)
    - A 2D tensor is like a matrix.
    - A matrix must be rectangular: every row has the same number of columns.
    - If one row has 5 elements and another has 4, PyTorch refuses.

    B) Mixed types (strings + numbers)
    - Tensors are numeric (used for math).
    - Mixing strings and numbers prevents PyTorch from picking a numeric dtype.
    - NumPy sometimes converts everything to strings, but that’s not useful for ML.

    This demo reproduces your errors exactly and explains them.
    """
    title("2) COMMON ERRORS: ragged rows and mixed types")

    # A) Ragged rows
    try:
        t3 = torch.tensor([[1, 2, 3, 4],
                           [5, 6, 7, 8, 111],
                           [9, 10, 11, 12]])
        show_io("Input (ragged list)", "row lengths: 4, 5, 4", "Output", t3)
    except Exception as e:
        show_io("Input (ragged list)", "row lengths: 4, 5, 4", "Output (error)", e)

    print("\nWhy this error happens:")
    print("- PyTorch tries to build a 3×4 tensor, but the second row is length 5.")
    print("- There is no single rectangular shape that fits all rows.")

    # B) Mixed types
    a = [["aa", 1, 2, 3],
         [5, 6, 7, 8],
         [9, 10, 11, 12]]
    try:
        ta = torch.tensor(a)
        show_io("Input (mixed types)", a, "Output", ta)
    except Exception as e:
        show_io("Input (mixed types)", a, "Output (error)", e)

    print("\nWhy this error happens:")
    print("- A tensor must have ONE dtype (example: float32).")
    print("- A string 'aa' cannot be stored in a numeric tensor.")


# =========================================================
# 3) RESHAPING
# =========================================================
def demo_reshaping() -> None:
    """
    Reshaping changes how the SAME elements are grouped into dimensions.

    Key rules:
    1) Reshape cannot change the number of elements.
       Example: 3×4 = 12 elements can become 2×6 = 12 elements.
    2) reshape() is NOT in-place.
       It returns a new tensor with a new view of the data.
       The old tensor stays the same unless you reassign.

    This demo shows:
    - Original tensor shape (3×4)
    - Reshaped tensor shape (2×6)
    - Original tensor remains (3×4)
    """
    title("3) RESHAPING: reshape() returns a new tensor")

    t2 = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

    show_io("Original t2", t2, "Original t2.shape", t2.shape)

    t2r = t2.reshape(2, 6)
    show_io("Input", "t2.reshape(2,6)", "Output (reshaped)", t2r)
    show_io("t2r.shape", t2r.shape, "Original t2.shape (unchanged)", t2.shape)

    print("\nKey idea:")
    print("- reshape creates a new object (t2r).")
    print("- If you want to keep it, you must do: t2 = t2.reshape(2,6)")


# =========================================================
# 4) DEVICES (CPU vs GPU / CUDA)
# =========================================================
def demo_devices() -> torch.device:
    """
    Devices: where tensors live and where computations happen.

    Device options:
    - cpu  : always available, runs on the processor
    - cuda : runs on an NVIDIA GPU (only if you have CUDA available)

    Why it matters:
    - GPUs are much faster for large matrix operations (deep learning training).
    - But CUDA is not always available, so code should fall back to CPU.

    Best practice:
    - Pick your device once:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    - Move tensors or models to that device.

    This demo shows:
    - Checking CUDA availability
    - Selecting a device
    - Moving a tensor to the chosen device
    """
    title("4) DEVICES: CPU vs GPU (CUDA)")

    cuda_ok = torch.cuda.is_available()
    show_io("torch.cuda.is_available()", cuda_ok, "Meaning", "True => CUDA usable, False => CPU only")

    device = torch.device("cuda") if cuda_ok else torch.device("cpu")
    show_io("Selected device", device, "Why we do this", "Same code works on CPU or GPU")

    t = torch.rand(2, 3)
    show_io("Random tensor device (default)", t.device, "After .to(device)", t.to(device).device)

    return device


# =========================================================
# 5) AUTOGRAD + GRADIENT DESCENT (Concept central)
# =========================================================
def demo_autograd_and_training_link() -> None:
    """
    AUTOGRAD + GRADIENT DESCENT — The Core Idea of Deep Learning

    This section connects TWO fundamental ideas:

    1) Autograd (automatic differentiation)
       - Computes derivatives automatically.
       - Stores gradients in tensor.grad after backward().

    2) Gradient Descent (learning rule)
       - Uses those gradients to update model parameters.
       - Parameters are moved in the direction that reduces the loss.

    KEY CONCEPT:
        Gradient = slope
        Learning = moving opposite to the slope

    ---------------------------------------------------------
    PART A — Pure Autograd (mathematical example)
    ---------------------------------------------------------
    We compute:
        e = (a*b - 5)^2

    Then:
        de/da
        de/db

    This shows how PyTorch applies the chain rule automatically.
    ---------------------------------------------------------
    """

    title("5A) AUTOGRAD: automatic gradient computation")

    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(4.0, requires_grad=True)

    c = a * b
    d = c - 5
    e = d ** 2

    print(f"a = {a.item()}, b = {b.item()}")
    print(f"e = (a*b - 5)^2 = {e.item()}")

    e.backward()

    print("\nComputed gradients:")
    print(f"de/da = {a.grad.item()}")
    print(f"de/db = {b.grad.item()}")

    print("\nInterpretation:")
    print("- Gradient tells how much e changes if we change a or b slightly.")
    print("- PyTorch used the chain rule automatically.")


    """
    ---------------------------------------------------------
    PART B — Using gradients to learn (training example)
    ---------------------------------------------------------
    Definitions (before training example)
    
    y_pred :
        The predicted value produced by the model.
        Here: y_pred = w * x
        It is what the model THINKS the output should be.
    
    MSE (Mean Squared Error) :
        The loss function measuring the difference between
        prediction and true value.
        MSE = mean((y_pred - y)^2)
        It tells us how wrong the model is.
    
    w :
        The model parameter (weight).
        This is what we are trying to learn.
    
    Gradient :
        The derivative of the loss with respect to a parameter.
        It tells us:
            - In which direction to change the parameter
            - How strongly to change it
    
    dLoss :
        The loss value (scalar).
        Represents total error.
    
    dL/dw (dw) :
        The derivative of the loss with respect to w.
        Also written as:
            gradient = dLoss/dw
        It tells us how sensitive the loss is to changes in w.
    
    learning_rate :
        A small positive number controlling how big each update step is.
        Too big  -> unstable training
        Too small -> very slow learning
    
    w.grad :
        In PyTorch, this stores dLoss/dw automatically
        after calling loss.backward().
        It contains the computed gradient.
    ---------------------------------------------------------
    
    Now we connect gradients to learning.

    Model:
        y_pred = w * x

    Loss:
        MSE = mean((y_pred - y)^2)

    Gradient:
        dLoss/dw

    Update rule:
        w = w - learning_rate * gradient

    This is Gradient Descent.
    ---------------------------------------------------------
    """

    title("5B) GRADIENT DESCENT: using gradients to learn")

    # Simple dataset: approximately y = 2x
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    w = torch.tensor([1.0], requires_grad=True)
    lr = 0.1

    for epoch in range(5):
        # Forward pass
        y_pred = w * x

        # Loss
        loss = ((y_pred - y) ** 2).mean()

        # Backward pass
        loss.backward()

        print(f"\nEpoch {epoch}")
        print(f"  w       = {w.item():.4f}")
        print(f"  loss    = {loss.item():.4f}")
        print(f"  w.grad  = {w.grad.item():.4f}")

        # Update (VERY IMPORTANT: disable gradient tracking)
        with torch.no_grad():
            w -= lr * w.grad

        # Clear gradient (VERY IMPORTANT)
        w.grad.zero_()

    print("\nFinal understanding:")
    print("- Autograd computes the slope.")
    print("- Gradient descent uses that slope to improve parameters.")
    print("- This loop is the foundation of ALL deep learning models.")

# =========================================================
# 6) TRAINING (mini loop)
# =========================================================
def demo_training(device: torch.device) -> Tuple[nn.Module, optim.Optimizer, List[float], List[float]]:
    """
    GOAL OF THIS SECTION
    --------------------
    Teach the core training logic used in *all* deep learning:

        Forward  -> compute predictions
        Loss     -> measure error
        Backward -> compute gradients (slopes)
        Update   -> change parameters to reduce the loss

    We train a tiny model:
        y_pred = w * x
    on a small dataset where y is approximately 2*x.

    We show TWO versions:
    A) Manual gradient descent (you update w yourself)
    B) PyTorch style: nn.Module + optimizer (standard professional workflow)

    Returns:
        model, optimizer, losses, weights (useful for plotting and checkpointing)
    """
    title("6) TRAINING: manual loop and nn.Module + optimizer")

    # ---------------------------------------------------------
    # Dataset (x -> y)
    # ---------------------------------------------------------
    # EXPECTED: x and y are float tensors (float32) so gradients can be computed.
    # They are moved to the chosen device (CPU/GPU).
    x = torch.tensor([[7.01], [3.02], [4.99], [8.00]], dtype=torch.float32, device=device)
    y = torch.tensor([[14.01], [6.01], [10.00], [16.04]], dtype=torch.float32, device=device)

    # =========================================================
    # A) Manual gradient descent
    # =========================================================
    title("6A) TRAINING (manual): y_pred = w*x")

    # w is the ONLY learnable parameter here.
    # requires_grad=True means: PyTorch will compute d(loss)/d(w) after backward().
    w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True, device=device)

    lr = 0.001  # learning rate: step size for parameter updates

    for epoch in range(10):
        # 1) FORWARD PASS
        # ----------------
        # Compute model predictions using the current parameter w.
        # Here the model is: y_pred = w*x
        y_pred = w * x

        # 2) LOSS (error measure)
        # -----------------------
        # Mean Squared Error:
        #   loss = mean((y_pred - y)^2)
        # This is a single scalar number.
        loss = ((y_pred - y) ** 2).mean()

        # 3) BACKWARD PASS (gradients)
        # ----------------------------
        # Compute gradient of loss w.r.t. w:
        #   w.grad = d(loss)/d(w)
        # This uses automatic differentiation (chain rule).
        loss.backward()

        # (Optional but very educational) See the gradient value:
        grad_w = w.grad.item()

        # 4) UPDATE STEP (gradient descent)
        # --------------------------------
        # We want to REDUCE the loss, so we move w opposite to the gradient:
        #   w = w - lr * w.grad
        #
        # Important: we must NOT track this update in Autograd.
        # That's why we use torch.no_grad().
        with torch.no_grad():
            w -= lr * w.grad

            # IMPORTANT: gradients accumulate by default in PyTorch.
            # If we do not clear them, next epoch will add new gradients to old ones.
            w.grad.zero_()

        print(f"Epoch {epoch:02d}: Loss={loss.item():.6f}, w={w.item():.6f}, grad={grad_w:.6f}")

    # =========================================================
    # B) nn.Module + optimizer (standard PyTorch)
    # =========================================================
    title("6B) TRAINING (PyTorch style): nn.Module + optimizer")

    """
    WHAT nn.Module DOES (high-level)
    -------------------------------
    nn.Module is the standard base class for models in PyTorch.

    It organizes:
    - Learnable parameters (weights) as nn.Parameter
    - The forward computation (forward method)
    - device moves: model.to(device)
    - saving/loading: model.state_dict()
    - integration with optimizers: optimizer = SGD(model.parameters(), ...)

    Key idea:
        forward() defines the math
        optimizer.step() updates the parameters using the gradients
    """

    class MyMulModel(nn.Module):
        """
        Minimal model:
            y_pred = w * x

        - self.w is a learnable weight (nn.Parameter)
        - forward(x_in) defines the computation
        """
        def __init__(self):
            super().__init__()

            # nn.Parameter registers the tensor as a trainable parameter:
            # - requires_grad=True automatically
            # - appears in model.parameters() so the optimizer can update it
            self.w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        def forward(self, x_in):
            # Forward pass: compute prediction.
            # When we call model(x), PyTorch actually calls forward(x) here.
            return self.w * x_in

    # Instantiate model and move it to the same device as the data
    model = MyMulModel().to(device)

    # Show what's inside (educational):
    print("\nModel:", model)
    print("Parameters:", [name for name, _ in model.named_parameters()])
    print("state_dict keys:", list(model.state_dict().keys()))

    # Loss function object (same MSE as manual version but packaged nicely)
    criterion = nn.MSELoss()

    # Optimizer knows which parameters to update because we pass model.parameters()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    losses: List[float] = []
    weights: List[float] = []

    for epoch in range(10):
        # 1) FORWARD PASS
        # ----------------
        # model(x) calls forward(x) and builds the computation graph.
        y_pred = model(x)

        # 2) LOSS
        # --------
        # criterion(y_pred, y) computes the scalar loss.
        loss = criterion(y_pred, y)

        # 3) BACKWARD PASS
        # -----------------
        # Reset gradients (otherwise they accumulate).
        optimizer.zero_grad()

        # Compute gradients for ALL model parameters.
        # Here it computes: model.w.grad
        loss.backward()

        # Educational: prove what step() will do
        w_before = model.w.item()
        grad = model.w.grad.item()
        lr = optimizer.param_groups[0]["lr"]
        expected_w = w_before - lr * grad

        # 4) UPDATE STEP
        # --------------
        # optimizer.step() updates each parameter (here only w) using its gradient.
        # For SGD without momentum, it effectively does:
        #   w = w - lr * w.grad
        optimizer.step()

        w_after = model.w.item()

        losses.append(loss.item())
        weights.append(w_after)

        print(
            f"Epoch {epoch:02d}: "
            f"Loss={loss.item():.6f}, "
            f"w_before={w_before:.6f}, grad={grad:.6f}, expected_w={expected_w:.6f}, w_after={w_after:.6f}"
        )

    return model, optimizer, losses, weights

# =========================================================
# 7) CHECKPOINTS + TRACKING
# =========================================================
def demo_checkpoints_and_tracking(
    model: nn.Module,
    optimizer: optim.Optimizer,
    losses: List[float],
    weights: List[float],
    device: torch.device,
    path: str = "mycheckpoint.pth",
) -> None:
    """
    CHECKPOINTS (saving & loading)

    What is a checkpoint?
    - A checkpoint is a saved snapshot of training so you can restart later.
    - Think: “Save game” in a video game.

    Why we save:
    - To resume training if it stops
    - To reuse the trained model later for predictions (inference)

    What we usually save:
    - model_state_dict: the learned weights
    - optimizer_state_dict: optimizer internal state (important to resume smoothly)
    - epoch: where we stopped
    - loss: last loss (optional)

    TRACKING TRAINING

    What is tracking?
    - Tracking means storing numbers (loss, weights) at each epoch.
    - This helps you see if training improves over time.

    Tools:
    - Matplotlib: plot loss curves
    - TensorBoard: real-time dashboards and experiment comparison

    This demo:
    - Saves a checkpoint
    - Loads it back
    - Restores model/optimizer states
    - Prints code snippets for Matplotlib and TensorBoard
    """
    title("7) CHECKPOINTS + TRACKING: save/load + visualize metrics")

    # ---- Save
    torch.save(
        {
            "epoch": len(losses) - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses[-1],
        },
        path,
    )
    show_io("Saved checkpoint path", path, "Saved keys", ["epoch", "model_state_dict", "optimizer_state_dict", "loss"])

    # ---- Load
    chkpnt = torch.load(path, map_location=device)
    show_io("Loaded checkpoint keys", list(chkpnt.keys()), "Loaded epoch", chkpnt["epoch"])

    # Restore into a new model/optimizer
    class MyMulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        def forward(self, x):
            return self.w * x

    restored_model = MyMulModel().to(device)
    restored_optimizer = optim.SGD(restored_model.parameters(), lr=0.001)

    restored_model.load_state_dict(chkpnt["model_state_dict"])
    restored_optimizer.load_state_dict(chkpnt["optimizer_state_dict"])

    show_io("Restored w", restored_model.w.item(), "Last saved loss", chkpnt["loss"])

    print("\nTracking (Matplotlib) snippet:")
    print("""
# import matplotlib.pyplot as plt
# plt.plot(losses, label="Loss")
# plt.plot(weights, label="Weight (w)")
# plt.legend()
# plt.show()
""")

    print("Tracking (TensorBoard) snippet:")
    print("""
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/MyMulModel_experiment")
# for epoch, (l, w) in enumerate(zip(losses, weights)):
#     writer.add_scalar("Loss", l, epoch)
#     writer.add_scalar("Weight/w", w, epoch)
# writer.close()
# tensorboard --logdir runs
""")

# =========================================================
# 8) VISUALIZE LEARNING PROCESS (Loss + Weight curves)
# =========================================================
def demo_visualize_learning(losses: List[float], weights: List[float]) -> None:
    """
    Visualize the learning process, like on the slide.

    We stored during training:
        - losses  : loss value at each epoch
        - weights : current value of w at each epoch

    EXPECTED:
    - Loss curve should go DOWN as training improves
    - Weight curve should move toward the best value (≈ 2 for this dataset)
    """

    import matplotlib.pyplot as plt

    # Create a wide figure with 2 plots side-by-side
    plt.figure(figsize=(12, 5))

    # -------------------------
    # Plot 1: Loss over epochs
    # -------------------------
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # -------------------------
    # Plot 2: Weight over epochs
    # -------------------------
    plt.subplot(1, 2, 2)
    plt.plot(weights, label="Weight (w)")
    plt.xlabel("Epoch")
    plt.ylabel("w")
    plt.title("Weight (w) over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()


"""
---------------------------------------------------------
PART 9 — Linear Regression with Batch Gradient Descent
(Explanation of the slide)
---------------------------------------------------------

This slide shows a complete implementation of Linear Regression
trained using Batch Gradient Descent (BGD).

1) The dataset

galaxy_data = np.array([[2,70],
                        [3,110],
                        [4,165],
                        [6,390],
                        [7,550]])

Each row contains:
    column 0 -> x value (feature)
    column 1 -> y value (target)

Here:
    x = phone model version (or galaxy index)
    y = price

2) The model

We assume a linear model:

    y_pred = w * x + b

where:
    w = slope (weight)
    b = bias (intercept)

We start with:
    w = 0
    b = 0

3) Learning rate

alpha = 0.01

This controls how big each update step is.
Too large -> unstable
Too small -> slow learning

4) Batch Gradient Descent loop

for iteration in range(10000):

At each iteration:

a) Compute gradients

gradient_b = mean((w*x + b - y))
gradient_w = mean(x * (w*x + b - y))

These are the derivatives of the MSE loss with respect to:
    - b
    - w

Because this is Batch Gradient Descent,
we compute the mean over ALL data points.

5) Parameter update

b -= alpha * gradient_b
w -= alpha * gradient_w

This moves parameters in the direction
that reduces the loss.

6) Monitoring training

Every 200 iterations,
the code prints:
    iteration number
    gradient values
    current w and b

This allows us to see convergence.

7) Final prediction

After training:
    print("Estimated price for Galaxy S5:", w*5 + b)

We use the learned linear model
to predict the price for x = 5.

---------------------------------------------------------
Summary
---------------------------------------------------------

The slide demonstrates:
- Linear model: y = wx + b
- MSE loss
- Gradient computation
- Batch Gradient Descent
- Parameter updates
- Final prediction

This is a complete classical ML pipeline
implemented from scratch using NumPy.
"""

# ---------------------------------------------------------
# 1) Dataset
# ---------------------------------------------------------
# Each row: [x, y]
galaxy_data = np.array([
    [2, 70],
    [3, 110],
    [4, 165],
    [6, 390],
    [7, 550]
])

x = galaxy_data[:, 0]
y = galaxy_data[:, 1]

# ---------------------------------------------------------
# 2) Initialize parameters
# ---------------------------------------------------------
w = 0.0   # slope
b = 0.0   # bias

# Learning rate
alpha = 0.01

# Number of iterations
num_iterations = 10000

# ---------------------------------------------------------
# 3) Training loop (Batch Gradient Descent)
# ---------------------------------------------------------
for iteration in range(num_iterations):

    # Forward pass
    y_pred = w * x + b

    # Compute gradients (derivatives of MSE)
    gradient_w = np.mean(x * (y_pred - y))
    gradient_b = np.mean(y_pred - y)

    # Update parameters
    w -= alpha * gradient_w
    b -= alpha * gradient_b

    # Print progress every 200 iterations
    if iteration % 200 == 0:
        loss = np.mean((y_pred - y) ** 2)
        print(f"it={iteration}, loss={loss:.2f}, w={w:.3f}, b={b:.3f}")

# ---------------------------------------------------------
# 4) Final model
# ---------------------------------------------------------
print("\nTraining finished.")
print(f"Final parameters: w = {w:.3f}, b = {b:.3f}")

# Predict price for x = 5
prediction = w * 5 + b
print(f"Estimated price for Galaxy S5 (x=5): {prediction:.2f}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:

    demo_tensors()
    demo_tensor_creation_errors()
    demo_reshaping()
    device = demo_devices()
    demo_autograd_and_training_link()
    model, optimizer, losses, weights = demo_training(device)
    demo_visualize_learning(losses, weights)
    demo_checkpoints_and_tracking(model, optimizer, losses, weights, device)

    title("Done ✅")
    print("PyTorch demo finished.")


if __name__ == "__main__":
    main()