import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the fixed integrand function
def integrand(rho, beta, c):
    numerator = c**2 * rho**2
    denominator = beta**2 * (beta**2 - rho**2)
    
    # Ensure denominator is positive to avoid invalid sqrt
    valid_mask = denominator > 0
    result = np.zeros_like(rho)
    
    # Compute only for valid values
    result[valid_mask] = 2 * np.pi * rho[valid_mask] * np.sqrt(1 + numerator[valid_mask] / denominator[valid_mask])
    
    return result

# Exact surface area function
def exact_surface_area(beta, c):
    e = 1 - beta**2 / c**2
    return 2 * np.pi * beta**2 * (1 + (c / beta) * np.arcsin(e))

# Box–Muller Transform to generate Gaussian-distributed samples
def box_muller_transform(N, mu=0, sigma=1):
    U1 = np.random.uniform(0, 1, N)
    U2 = np.random.uniform(0, 1, N)
    
    Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    
    return mu + sigma * Z1  # Use only Z1 for N(μ,σ)

# Monte Carlo Integration using Gaussian Proposal Function
def monte_carlo_gaussian(beta, c, N, mu=0, sigma=1):
    rho_samples = box_muller_transform(N, mu, sigma)
    
    # Compute importance weights (for reweighting due to Gaussian sampling)
    weights = np.exp((rho_samples - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    
    integral_estimate = np.mean(integrand(rho_samples, beta, c) * weights)
    return 2 * integral_estimate

# Parameters
beta = 1
c = 1  # Given: 2β = c = 1
N_values = [10, 100, 1000, 10000, 100000]  # Different sample sizes

# Compute Monte Carlo integration errors using Gaussian sampling
errors_gaussian = []
exact_area = exact_surface_area(beta, c)

for N in N_values:
    mc_gaussian = monte_carlo_gaussian(beta, c, N)
    error = abs(mc_gaussian - exact_area) / exact_area  # Relative error
    errors_gaussian.append(error)

# Plot histogram of errors for different sample sizes (log scale)
plt.figure(figsize=(8, 5))
plt.bar([str(N) for N in N_values], errors_gaussian, color='b', alpha=0.7)
plt.yscale("log")  # Log scale for better visualization
plt.xlabel("Number of Samples (N)")
plt.ylabel("Relative Error (log scale)")
plt.title("Monte Carlo Integration Error using Gaussian Sampling")
plt.grid(axis='y', linestyle="--")
plt.show()
plt.savefig("plot_f.png")

# Display the computed errors
df_errors = pd.DataFrame({"N": N_values, "Relative Error": errors_gaussian})
print(df_errors)