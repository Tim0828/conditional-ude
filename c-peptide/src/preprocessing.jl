# Function to calculate KL divergence between two distributions
function kl_divergence(p, q; bins=50)
    # Create histogram bins
    min_val = min(minimum(p), minimum(q))
    max_val = max(maximum(p), maximum(q))
    bin_edges = range(min_val, max_val, length=bins + 1)

    # Calculate normalized histograms
    p_hist = fit(Histogram, p, bin_edges).weights
    q_hist = fit(Histogram, q, bin_edges).weights

    # Normalize to probability distributions
    p_norm = p_hist ./ sum(p_hist)
    q_norm = q_hist ./ sum(q_hist)

    # Add small epsilon to avoid log(0)
    ε = 1e-10
    p_norm = p_norm .+ ε
    q_norm = q_norm .+ ε

    # Calculate KL divergence
    return sum(p_norm .* log.(p_norm ./ q_norm))
end

# Function to calculate total KL divergence for all metrics
function total_kl_divergence(train_indices, test_indices, metrics)
    total_kl = 0.0
    for metric in metrics
        kl_div = kl_divergence(metric[train_indices], metric[test_indices])
        total_kl += kl_div
    end
    return total_kl
end

# Function to optimize train/test split to minimize KL divergence
function optimize_train_test_split(types, metrics, f_train, rng; n_attempts=1000)
    """
    Optimize train/test split to minimize KL divergence between train and test distributions.
    
    Args:
        types: Vector of type labels for stratified sampling
        metrics: Vector of metric arrays to minimize KL divergence for
        f_train: Fraction of data to use for training
        rng: Random number generator
        n_attempts: Number of random splits to try (default: 1000)
    
    Returns:
        Tuple of (train_indices, test_indices) with minimum KL divergence
    """
    best_kl = Inf
    best_train_indices = Int[]
    best_test_indices = Int[]

    # Try multiple random splits to find the one with minimum KL divergence
    for attempt in 1:n_attempts
        temp_train_indices, temp_test_indices = stratified_split(rng, types, f_train)

        # Calculate total KL divergence for this split
        kl_total = total_kl_divergence(temp_train_indices, temp_test_indices, metrics)

        if kl_total < best_kl
            best_kl = kl_total
            best_train_indices = temp_train_indices
            best_test_indices = temp_test_indices
        end
    end

    println("Best KL divergence: $best_kl")
    return best_train_indices, best_test_indices
end