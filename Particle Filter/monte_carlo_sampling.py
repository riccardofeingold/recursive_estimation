import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Samples
np.random.seed(42)  # For reproducibility
n_samples = 10000  # Number of samples
samples = np.random.normal(0, 1, n_samples)  # Generate samples from N(0, 1)

# Step 2: Estimate PDF using simple binning method
def estimate_pdf(samples, num_bins=50):
    min_val = min(samples)
    max_val = max(samples)
    bin_width = (max_val - min_val) / num_bins
    
    # Create bins
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    # Count samples in each bin
    bin_counts = np.zeros(num_bins)
    for sample in samples:
        bin_index = int((sample - min_val) / bin_width)
        if bin_index >= num_bins:
            bin_index = num_bins - 1
        bin_counts[bin_index] += 1
    
    # Normalize to form a PDF
    pdf_estimate = bin_counts / (n_samples * bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, pdf_estimate

# Estimate the PDF
bin_centers, pdf_estimate = estimate_pdf(samples)

# Step 3: Plot the results
x = np.linspace(-5, 5, 1000)
true_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

plt.figure(figsize=(10, 6))
plt.plot(bin_centers, pdf_estimate, label='Estimated PDF', alpha=0.6)
plt.plot(x, true_pdf, label='True PDF (N(0,1))', linestyle='--')
plt.title('PDF Estimation using Monte Carlo Sampling')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
