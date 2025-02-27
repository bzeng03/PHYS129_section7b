import numpy as np
import scipy.integrate as spi
import pandas as pd
import matplotlib.pyplot as plt

# Define the function inside the integral
def integrand(rho, beta, c):
    numerator = c**2 * rho**2
    denominator = beta**2 * (beta**2 - rho**2)
    return 2 * np.pi * rho * np.sqrt(1 + numerator / denominator)

# Midpoint Rule for Numerical Integration
def midpoint_rule(beta, c, N=100):
    rho_vals = np.linspace(0, beta, N)
    midpoints = (rho_vals[:-1] + rho_vals[1:]) / 2
    delta_rho = beta / N
    integral_sum = np.sum(integrand(midpoints, beta, c)) * delta_rho
    return 2 * integral_sum

# Gaussian Quadrature
def gaussian_quadrature(beta, c, N=10):
    result, _ = spi.quadrature(lambda rho: integrand(rho, beta, c), 0, beta, maxiter=N)
    return 2 * result

# Exact formula for surface area
def exact_surface_area(beta, c):
    e = 1 - beta**2 / c**2
    return 2 * np.pi * beta**2 * (1 + (c / beta) * np.arcsin(e))

# Define beta and c values
beta_vals = np.logspace(-3, 3, 50)  # Beta values from 0.001 to 1000
c_vals = np.logspace(-3, 3, 50)  # C values from 0.001 to 1000

# Initialize error matrices
midpoint_errors = np.zeros((len(beta_vals), len(c_vals)))
gaussian_errors = np.zeros((len(beta_vals), len(c_vals)))

# Compute errors
for i, beta in enumerate(beta_vals):
    for j, c in enumerate(c_vals):
        if c > beta:  # Ensure valid ellipsoid condition
            exact = exact_surface_area(beta, c)
            midpoint = midpoint_rule(beta, c)
            gaussian = gaussian_quadrature(beta, c)

            midpoint_errors[i, j] = abs(midpoint - exact) / exact
            gaussian_errors[i, j] = abs(gaussian - exact) / exact
        else:
            midpoint_errors[i, j] = np.nan
            gaussian_errors[i, j] = np.nan

# Convert errors to DataFrames
df_midpoint_errors = pd.DataFrame(midpoint_errors, index=beta_vals, columns=c_vals)
df_gaussian_errors = pd.DataFrame(gaussian_errors, index=beta_vals, columns=c_vals)

# Plot error heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Midpoint Error Heatmap
im1 = axes[0].imshow(midpoint_errors, extent=[c_vals.min(), c_vals.max(), beta_vals.min(), beta_vals.max()], 
                      origin='lower', aspect='auto', cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
axes[0].set_title("Midpoint Rule Error")
axes[0].set_xlabel("c values")
axes[0].set_ylabel("β values")
fig.colorbar(im1, ax=axes[0])

# Gaussian Quadrature Error Heatmap
im2 = axes[1].imshow(gaussian_errors, extent=[c_vals.min(), c_vals.max(), beta_vals.min(), beta_vals.max()], 
                      origin='lower', aspect='auto', cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
axes[1].set_title("Gaussian Quadrature Error")
axes[1].set_xlabel("c values")
axes[1].set_ylabel("β values")
fig.colorbar(im2, ax=axes[1])

plt.show()
plt.savefig("plot_b.png")