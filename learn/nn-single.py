"""
The smallest possible neural network: ONE neuron learning AND.

    z = w1*x1 + w2*x2 + b            weighted sum
    a = 1 / (1 + e^-z)               sigmoid -> prediction in (0, 1)
    L = -[y*ln(a) + (1-y)*ln(1-a)]   cross-entropy loss for one point

    dL/dz  = a - y                   (sigmoid + cross-entropy simplify to this)
    dL/dw1 = dz*x1   dL/dw2 = dz*x2   dL/db = dz

    w -= learning_rate * (average gradient over the 4 points)

Run:  python minimal.py
"""
import math

# ---------------------------------------------------------------- configuration
learning_rate = 0.5
epochs = 5

# AND dataset: (x1, x2) -> y
points = [(0, 0), (0, 1), (1, 0), (1, 1)]
targets = [0, 0, 0, 1]

# fixed starting values (so every run prints the same numbers)
w1 = 0.5
w2 = -0.3
b = 0.0

# ---------------------------------------------------------------- training loop
for epoch in range(1, epochs + 1):
    print("=" * 74)
    print(f"EPOCH {epoch}   w1={w1:+.4f}  w2={w2:+.4f}  b={b:+.4f}")
    print("=" * 74)

    sum_dw1 = 0.0
    sum_dw2 = 0.0
    sum_db = 0.0
    sum_loss = 0.0

    for (x1, x2), y in zip(points, targets):
        # forward pass
        z = w1 * x1 + w2 * x2 + b
        a = 1 / (1 + math.exp(-z))
        loss = -(y * math.log(a) + (1 - y) * math.log(1 - a))

        # backward pass (gradients for this one point)
        dz = a - y
        dw1 = dz * x1
        dw2 = dz * x2
        db = dz

        print(f"point ({x1},{x2})  target y={y}")
        print(f"  forward : z = {w1:+.3f}*{x1} + {w2:+.3f}*{x2} + {b:+.3f} = {z:+.4f}"
              f"   a = {a:.4f}   loss = {loss:.4f}")
        print(f"  backward: dz = a - y = {dz:+.4f}   dw1 = {dw1:+.4f}"
              f"   dw2 = {dw2:+.4f}   db = {db:+.4f}")

        sum_dw1 += dw1
        sum_dw2 += dw2
        sum_db += db
        sum_loss += loss

    # average over the 4 points
    n = len(points)
    g_w1 = sum_dw1 / n
    g_w2 = sum_dw2 / n
    g_b = sum_db / n
    print("-" * 74)
    print(f"mean loss = {sum_loss / n:.4f}")
    print(f"mean gradients: dw1 = {g_w1:+.4f}   dw2 = {g_w2:+.4f}   db = {g_b:+.4f}")

    # gradient descent update
    print(f"update: w1 = {w1:+.4f} - {learning_rate}*({g_w1:+.4f}) = {w1 - learning_rate * g_w1:+.4f}")
    print(f"        w2 = {w2:+.4f} - {learning_rate}*({g_w2:+.4f}) = {w2 - learning_rate * g_w2:+.4f}")
    print(f"        b  = {b:+.4f} - {learning_rate}*({g_b:+.4f}) = {b - learning_rate * g_b:+.4f}")
    w1 -= learning_rate * g_w1
    w2 -= learning_rate * g_w2
    b -= learning_rate * g_b

print("=" * 74)
print(f"final: w1={w1:+.4f}  w2={w2:+.4f}  b={b:+.4f}")