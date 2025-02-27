import numpy as np
import matplotlib.pyplot as plt

def van_der_corput(n, base=2):
    """Generates the n first elements of a Van der Corput sequence in the given base."""
    sequence = np.zeros(n)
    for i in range(n):
        num, denom = i, 1
        while num > 0:
            denom *= base
            num, remainder = divmod(num, base)
            sequence[i] += remainder / denom
    return sequence

def sobol_2d(n):
    """
    Generate the first `n` points of a 2D Sobol sequence using a standard approach.
    
    Parameters:
    - n: Number of points to generate
    
    Returns:
    - points: Array of shape (n, 2) containing the Sobol points
    """
    points = np.zeros((n, 2))
    
    # First dimension (van der Corput sequence in base 2)
    points[:, 0] = van_der_corput(n)
    
    # Second dimension using bitwise Sobol sequence approach
    x = 0
    direction = np.array([1 << (31 - i) for i in range(31)], dtype=np.uint32)  # Precomputed direction numbers
    
    for i in range(n):
        c = 0
        value = i
        while value & 1:
            c += 1
            value >>= 1
        
        x ^= direction[c]
        points[i, 1] = x / (1 << 32)  # Normalize to [0,1]
    
    return points

# Generate 50 points
points = sobol_2d(50)

# Plot the points
plt.figure(figsize=(10, 8))
plt.scatter(points[:, 0], points[:, 1], s=40, alpha=0.7)  # Reduced size to avoid clutter
plt.grid(True, alpha=0.3)
plt.title('First 50 points of 2D Sobol Sequence')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('sobol_sequence.png')
plt.show()

# Print all 50 points
print("First 50 points of the Sobol sequence:")
for i in range(50):
    print(f"Point {i+1:2d}: ({points[i, 0]:.6f}, {points[i, 1]:.6f})")