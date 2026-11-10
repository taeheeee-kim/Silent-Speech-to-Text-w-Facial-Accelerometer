import numpy as np

def phoneme_error_rate(pred_seq, target_seq):
    """
    Compute Phoneme Error Rate (PER) between predicted and target sequences.
    - pred_seq, target_seq: list of phoneme indices or symbols
    """
    n = len(target_seq)
    m = len(pred_seq)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if target_seq[i - 1] == pred_seq[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # deletion
                dp[i, j - 1] + 1,      # insertion
                dp[i - 1, j - 1] + cost  # substitution
            )

    dist = dp[n, m]
    return dist / max(1, n)  # PER = edit_distance / target_length
