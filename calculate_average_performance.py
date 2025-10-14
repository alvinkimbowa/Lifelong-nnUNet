#!/usr/bin/env python3
"""
Script to calculate average performance across all folds for NCA training results.
"""

import csv
import os
from pathlib import Path

def calculate_fold_averages(results_dir):
    """Calculate average performance metrics across all folds."""
    
    # Find all fold directories
    fold_dirs = []
    for item in os.listdir(results_dir):
        if item.startswith('fold_') and os.path.isdir(os.path.join(results_dir, item)):
            fold_dirs.append(item)
    
    print(f"Found fold directories: {fold_dirs}")
    
    all_metrics = []
    
    for fold_dir in sorted(fold_dirs):
        fold_path = os.path.join(results_dir, fold_dir)
        csv_file = os.path.join(fold_path, 'val_metrics.csv')
        
        if os.path.exists(csv_file):
            print(f"Processing {fold_dir}...")
            
            # Read CSV file manually
            dice_values = []
            iou_values = []
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    # Check if this is the final epoch
                    if row.get('Epoch') == 'epoch_999':
                        metric = row.get('metric', '')
                        value = float(row.get('value', 0))
                        
                        if metric == 'Dice':
                            dice_values.append(value * 100)
                        elif metric == 'IoU':
                            iou_values.append(value * 100)
            
            if dice_values and iou_values:
                avg_dice = sum(dice_values) / len(dice_values)
                avg_iou = sum(iou_values) / len(iou_values)
                
                fold_metrics = {
                    'fold': fold_dir,
                    'Dice': avg_dice,
                    'IoU': avg_iou,
                    'num_samples': len(dice_values)
                }
                all_metrics.append(fold_metrics)
                
                print(f"  {fold_dir} - Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f} ({len(dice_values)} samples)")
            else:
                print(f"  {fold_dir} - No final epoch data found")
        else:
            print(f"  {fold_dir} - No val_metrics.csv found")
    
    if not all_metrics:
        print("No metrics found!")
        return None
    
    # Calculate overall averages
    print("\n" + "="*60)
    print("OVERALL AVERAGE PERFORMANCE ACROSS ALL FOLDS")
    print("="*60)
    
    dice_values = [m['Dice'] for m in all_metrics]
    iou_values = [m['IoU'] for m in all_metrics]
    
    for metric_name, values in [('Dice', dice_values), ('IoU', iou_values)]:
        mean_val = sum(values) / len(values)
        
        # Calculate standard deviation
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = variance ** 0.5
        
        min_val = min(values)
        max_val = max(values)
        
        print(f"{metric_name:>6}: {mean_val:.4f} ± {std_val:.4f} (range: {min_val:.4f} - {max_val:.4f})")
    
    # Show per-fold breakdown
    print("\n" + "="*60)
    print("PER-FOLD BREAKDOWN")
    print("="*60)
    print(f"{'Fold':<12} {'Dice':<8} {'IoU':<8} {'Samples':<8}")
    print("-" * 40)
    
    for metrics in all_metrics:
        print(f"{metrics['fold']:<12} {metrics['Dice']:<8.4f} {metrics['IoU']:<8.4f} {metrics['num_samples']:<8}")
    
    # Save results
    output_file = os.path.join(results_dir, 'fold_average_performance.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'Dice', 'IoU', 'num_samples'])
        for metrics in all_metrics:
            writer.writerow([metrics['fold'], metrics['Dice'], metrics['IoU'], metrics['num_samples']])
    
    print(f"\nResults saved to: {output_file}")
    
    return all_metrics

def main():
    # Path to your results directory
    results_dir = "data/nnUNet_results/nnUNet_ext/2d/Task073_GE_LE/Task073_GE_LE/nnUNetTrainerNCA__nnUNetPlansv2.1/Generic_UNet/SEQ"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    print("Calculating average performance across folds...")
    print(f"Results directory: {results_dir}")
    print()
    
    metrics_list = calculate_fold_averages(results_dir)
    
    if metrics_list is not None:
        print(f"\nTotal folds analyzed: {len(metrics_list)}")

if __name__ == "__main__":
    main()
