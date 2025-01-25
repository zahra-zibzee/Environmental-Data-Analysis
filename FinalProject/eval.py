import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance


def compute_pairwise_distances(true_coords, pred_coords):
    """
    Compute pairwise Euclidean distances between true and predicted coordinates.

    Parameters:
        true_coords (np.ndarray): Ground truth coordinates of shape (N, 2).
        pred_coords (np.ndarray): Predicted coordinates of shape (M, 2).

    Returns:
        np.ndarray: Distance matrix of shape (M, N).
    """
    return distance.cdist(pred_coords, true_coords, metric="euclidean")


def evaluate_with_threshold(true_coords, pred_coords, threshold=2.0):
    """
    Evaluate precision, recall, and F1-score using a distance threshold.

    Parameters:
        true_coords (np.ndarray): Ground truth coordinates of shape (N, 2).
        pred_coords (np.ndarray): Predicted coordinates of shape (M, 2).
        threshold (float): Maximum distance to consider a match.

    Returns:
        dict: Precision, recall, F1-score, and counts of TP, FP, FN.
    """
    dists = compute_pairwise_distances(true_coords, pred_coords)

    # Find closest matches for each predicted point
    matched_pred = np.min(dists, axis=1) < threshold
    tp = np.sum(matched_pred)  # True positives
    fp = len(pred_coords) - tp  # False positives

    # Find unmatched ground truth points
    matched_true = np.min(dists, axis=0) < threshold
    fn = len(true_coords) - np.sum(matched_true)  # False negatives

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def evaluate_with_hungarian(true_coords, pred_coords, threshold=2.0):
    """
    Evaluate metrics using the Hungarian algorithm for optimal matching.

    Parameters:
        true_coords (np.ndarray): Ground truth coordinates of shape (N, 2).
        pred_coords (np.ndarray): Predicted coordinates of shape (M, 2).
        threshold (float): Maximum distance to consider a match.

    Returns:
        dict: Precision, recall, F1-score, unmatched predictions, and unmatched ground truth.
    """
    dists = compute_pairwise_distances(true_coords, pred_coords)

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(dists)
    matched_dists = dists[row_ind, col_ind]

    # Count matches below the threshold
    matches = matched_dists < threshold
    tp = np.sum(matches)  # True positives
    fp = len(pred_coords) - tp  # False positives
    fn = len(true_coords) - tp  # False negatives

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "unmatched_predictions": len(pred_coords) - tp,
        "unmatched_ground_truth": len(true_coords) - tp,
    }


def compute_rmse(true_coords, pred_coords, *args):
    """
    Compute the Root Mean Squared Error (RMSE) between matched points.

    Parameters:
        true_coords (np.ndarray): Ground truth coordinates of shape (N, 2).
        pred_coords (np.ndarray): Predicted coordinates of shape (M, 2).

    Returns:
        float: RMSE value.
    """
    dists = compute_pairwise_distances(true_coords, pred_coords)

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(dists)
    matched_dists = dists[row_ind, col_ind]

    return np.sqrt(np.mean(matched_dists**2))


def compute_mae(true_coords, pred_coords, *args):
    """
    Compute the Mean Absolute Error (MAE) between matched points.

    Parameters:
        true_coords (np.ndarray): Ground truth coordinates of shape (N, 2).
        pred_coords (np.ndarray): Predicted coordinates of shape (M, 2).

    Returns:
        float: MAE value.
    """
    dists = compute_pairwise_distances(true_coords, pred_coords)

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(dists)
    matched_dists = dists[row_ind, col_ind]

    return np.mean(matched_dists)


# Example usage
if __name__ == "__main__":
    true_coords = np.array([[1, 1], [2, 2], [3, 3]])
    pred_coords = np.array([[1.1, 1.1], [3.1, 3.1], [5, 5]])

    print(
        "Threshold-based evaluation:", evaluate_with_threshold(true_coords, pred_coords)
    )
    print(
        "Hungarian algorithm evaluation:",
        evaluate_with_hungarian(true_coords, pred_coords),
    )
    print("RMSE:", compute_rmse(true_coords, pred_coords))
    print("MAE:", compute_mae(true_coords, pred_coords))
