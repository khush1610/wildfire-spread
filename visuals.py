import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from params import HParams
import dataset
from test import build_model
import constants  # For INPUT_FEATURES

# Initialize hyperparameters
hparams = HParams()

# Load test dataset (use "test" mode with shuffling)
test_dataset = dataset.make_dataset(hparams, "predict").shuffle(buffer_size=1847).prefetch(1)
model = build_model(hparams)

# Load trained weights
model_path = './model_tuned/final_model.h5'  # Local path
try:
    model.load_weights(model_path)
    print(f"Loaded model from: {model_path}")
except Exception as e:
    print(f"Error loading weights: {e}")

# Custom colormaps
from matplotlib.colors import ListedColormap
cmap_binary = ListedColormap(['#AFE1AF', '#FF0000'])  # Gray (0), Orange (1)
cmap_continuous = plt.cm.viridis  # For continuous features

# Feature list from constants.INPUT_FEATURES (12 features)
feature_names = constants.INPUT_FEATURES  # ['elevation', 'population', 'NDVI', 'PrevFireMask', 'pdsi', 'vs', 'pr', 'sph', 'tmmx', 'tmmn', 'th', 'vpd']
feature_indices = {name: idx for idx, name in enumerate(feature_names)}

# Function to calculate IoU (Intersection over Union)
def calculate_iou(y_true, y_pred, threshold):
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_binary)
    union = np.sum(np.logical_or(y_true, y_pred_binary))
    return intersection / union if union > 0 else 0

# Get multiple samples and predict, select top 2 with best IoU and ≥10 fire pixels
samples = []
for inputs, labels in test_dataset.take(10):  # Take 10 shuffled samples
    preds = model.predict(inputs, batch_size=inputs.shape[0])
    input_batch = inputs.numpy()
    ground_truth_batch = labels.numpy()
    predicted_batch = preds

    # Evaluate each sample
    best_samples = []
    for i in range(inputs.shape[0]):
        gt_fire = ground_truth_batch[i, :, :, 0]
        fire_count = np.sum(gt_fire > 0)  # Count fire pixels
        if fire_count >= 10:  # Only consider samples with at least 10 fire pixels
            pred = predicted_batch[i, :, :, 0]
            # Find optimal threshold for this sample (maximize IoU)
            thresholds = np.arange(0.1, 0.9, 0.1)
            iou_scores = [calculate_iou(gt_fire, pred, thresh) for thresh in thresholds]
            best_threshold = thresholds[np.argmax(iou_scores)]
            best_pred_binary = (pred > best_threshold).astype(np.float32)
            iou_score = calculate_iou(gt_fire, best_pred_binary, best_threshold)
            best_samples.append((i, iou_score, best_pred_binary, gt_fire))

    # Sort by IoU and take top 2
    best_samples.sort(key=lambda x: x[1], reverse=True)
    top_2_samples = best_samples[:2]
    if top_2_samples:
        for i, iou, pred_binary, gt in top_2_samples:
            samples.append((input_batch[i], gt, pred_binary))
    break

if not samples:
    print("No samples with at least 10 fire pixels and reasonable prediction found in 10 attempts.")
else:
    # Only 3 columns: PrevFireMask, FireMask (ground truth), Predicted
    num_rows = min(2, len(samples))  # Use available samples, max 2
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows),
                             constrained_layout=True, gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    column_titles = ['PrevFireMask', 'FireMask (GT)', 'Predicted']
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=10, pad=2)

    for row, (inputs_sample, ground_truth, predicted_binary) in enumerate(samples):
        input_batch = inputs_sample
        prev_mask = input_batch[:, :, feature_indices['PrevFireMask']]

        print(f"Sample {row + 1} - PrevFireMask min/max: {prev_mask.min()}/{prev_mask.max()}")
        print(f"Sample {row + 1} - GroundTruth min/max: {ground_truth.min()}/{ground_truth.max()}")
        print(f"Sample {row + 1} - Predicted min/max: {predicted_binary.min()}/{predicted_binary.max()}")

        axes[row, 0].imshow(prev_mask, cmap=cmap_binary, vmin=0, vmax=1)
        axes[row, 1].imshow(ground_truth, cmap=cmap_binary, vmin=0, vmax=1)
        axes[row, 2].imshow(predicted_binary, cmap=cmap_binary, vmin=0, vmax=1)

        for col in range(3):
            axes[row, col].axis('off')

    plt.savefig('./imgs/prev_fire_gt_pred_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
