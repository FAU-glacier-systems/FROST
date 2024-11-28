def systematic_sample(mean, std, n_samples):
    """
    Systematically sample points from a normal distribution.

    Args:
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        n_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Array of systematically sampled values.
    """
    # Define the range of sampling (±3 standard deviations)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # Generate equally spaced points in the range
    samples = np.linspace(lower_bound, upper_bound, n_samples)

    # Optionally, evaluate the PDF at the sampled points (for weighting or analysis)
    pdf_values = norm.pdf(samples, loc=mean, scale=std)

    return samples, pdf_values