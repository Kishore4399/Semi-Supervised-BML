import os
import glob
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from sklearn.metrics import jaccard_score, precision_score, recall_score
import ace_tools as tools

validation_dir = "/shqin/boneMarrowLession/data/validation"
test_dir = "/shqin/boneMarrowLession/data/test"
results_dir = "/shqin/boneMarrowLession/data/post_processing_results"
os.makedirs(results_dir, exist_ok=True)

def load_images_from_folder(folder):
    """Load images from the specified folder using glob."""
    images = []
    filenames = []
    for filepath in sorted(glob.glob(os.path.join(folder, "*.bmp"))):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(os.path.basename(filepath))
    return images, filenames

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def generate_difference_map(original, reconstructed):
    diff_map = np.clip(original - reconstructed, 0, None)
    return diff_map

def binarize_otsu(diff_map):
    threshold = threshold_otsu(diff_map)
    binary_mask = diff_map > threshold
    return binary_mask.astype(np.uint8)

def morphological_post_processing(binary_mask, min_object_size=20, min_hole_size=20):
    binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=min_object_size)
    binary_mask = remove_small_holes(binary_mask, area_threshold=min_hole_size)
    return binary_mask.astype(np.uint8)

if __name__ == "__main__":
    validation_images, _ = load_images_from_folder(validation_dir)
    validation_reconstructed, _ = load_images_from_folder(validation_dir + "_reconstructed")
    validation_gt, _ = load_images_from_folder(validation_dir + "_gt")

    best_dice_score = 0
    best_params = {}
    min_object_sizes = [20, 30, 40, 50, 60]
    min_hole_sizes = [20, 30, 40, 50, 60]

    for obj_size in min_object_sizes:
        for hole_size in min_hole_sizes:
            total_dice = 0
            for i in range(len(validation_images)):
                diff_map = generate_difference_map(histogram_equalization(validation_images[i]),
                                                   histogram_equalization(validation_reconstructed[i]))
                binary_mask = binarize_otsu(diff_map)
                refined_mask = morphological_post_processing(binary_mask, obj_size, hole_size)
                ground_truth = validation_gt[i]
                dice_score = (2 * np.logical_and(refined_mask, ground_truth).sum()) / (refined_mask.sum() + ground_truth.sum() + 1e-6)
                total_dice += dice_score
            avg_dice = total_dice / len(validation_images)
            if avg_dice > best_dice_score:
                best_dice_score = avg_dice
                best_params = {"min_object_size": obj_size, "min_hole_size": hole_size}

    # Load test data
    test_images, test_filenames = load_images_from_folder(test_dir)
    test_reconstructed, _ = load_images_from_folder(test_dir + "_reconstructed")
    test_gt, _ = load_images_from_folder(test_dir + "_gt")

    test_results = []
    for i in range(len(test_images)):
        diff_map = generate_difference_map(histogram_equalization(test_images[i]), histogram_equalization(test_reconstructed[i]))
        binary_mask = binarize_otsu(diff_map)
        final_mask = morphological_post_processing(binary_mask, best_params["min_object_size"], best_params["min_hole_size"])
        ground_truth = test_gt[i]
        
        dice_score = (2 * np.logical_and(final_mask, ground_truth).sum()) / (final_mask.sum() + ground_truth.sum() + 1e-6)
        iou_score = jaccard_score(ground_truth.flatten(), final_mask.flatten())
        precision = precision_score(ground_truth.flatten(), final_mask.flatten())
        recall = recall_score(ground_truth.flatten(), final_mask.flatten())
        
        test_results.append([test_filenames[i], dice_score, iou_score, precision, recall])
        
        cv2.imwrite(os.path.join(results_dir, f"{test_filenames[i]}_diff_map.bmp"), (diff_map / diff_map.max() * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(results_dir, f"{test_filenames[i]}_binary_mask.bmp"), (binary_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(results_dir, f"{test_filenames[i]}_final_mask.bmp"), (final_mask * 255).astype(np.uint8))

    # Convert results to DataFrame and display
    df_results = pd.DataFrame(test_results, columns=["Test Image", "Dice Score", "IoU", "Precision", "Recall"])
    tools.display_dataframe_to_user(name="Test Results", dataframe=df_results)
