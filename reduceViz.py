import numpy as np
import matplotlib.pyplot as plt

def l1_norm(x, y):
    return np.abs(x) + np.abs(y)

def l2_norm(x, y):
    return np.sqrt(x**2 + y**2)

def l_inf_norm(x, y):
    return np.maximum(np.abs(x), np.abs(y))

def mean(x, y):
    return (np.abs(x) + np.abs(y)) / 2

def plot_contours(ax, norm_function, title):
    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = norm_function(X, Y)

    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal', 'box')  # This ensures a 1:1 aspect ratio
    ax.grid(True)
    return contour

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle("Contour Plots of Different Norms", fontsize=16)

# Plot contours for different norms
norms = [
    (l1_norm, "L1 Norm (Manhattan)"),
    (l2_norm, "L2 Norm (Euclidean)"),
    (l_inf_norm, "L-infinity Norm"),
    (mean, "Mean")
]

for ax, (norm_func, title) in zip(axs.ravel(), norms):
    contour = plot_contours(ax, norm_func, title)
    fig.colorbar(contour, ax=ax, label='Norm value')

plt.tight_layout()
plt.show()