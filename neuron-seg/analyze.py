import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from collections import defaultdict
import seaborn as sns

def parse_evaluation_results(seg_path, sample_name='B'):
    """
    Parse evaluation results from segmentation output folders
    """
    results = defaultdict(dict)
    
    # SOTA results
    sota_path = os.path.join(seg_path, 'sota')
    if os.path.exists(sota_path):
        results['SOTA'] = parse_folder_results(sota_path)
    
    # CM-GLLF results
    Mus = list(np.around(np.arange(0.1, 1, 0.1), decimals=1))
    Is = list(np.around(np.arange(0.1, 1, 0.1), decimals=1))
    
    for mu in Mus:
        for inflec in Is:
            folder_name = f'mu_{mu}_I_{inflec}'
            folder_path = os.path.join(seg_path, folder_name)
            if os.path.exists(folder_path):
                results[folder_name] = parse_folder_results(folder_path)
    
    return results

def parse_folder_results(folder_path):
    """
    Parse results from a single folder (assumes evaluation outputs exist)
    """
    results = {}
    
    # Look for evaluation result files (you may need to adjust based on your evaluate function output)
    result_files = glob.glob(os.path.join(folder_path, '*th_*.json'))
    if not result_files:
        # If no JSON files, look for other result formats
        result_files = glob.glob(os.path.join(folder_path, '*th_*'))
    
    for file_path in result_files:
        try:
            # Extract threshold and merge function from filename
            filename = os.path.basename(file_path)
            # Parse filename like 'median_aff_histogramsth_0.5'
            parts = filename.split('th_')
            if len(parts) == 2:
                merge_func = parts[0]
                threshold = float('.'.join(filename.split('_')[-1].split('.json')[0].split('.')[-2:]))
                
                # Try to read JSON results
                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        results[(merge_func, threshold)] = data
                else:
                    # If not JSON, assume it's a text file with metrics
                    results[(merge_func, threshold)] = parse_text_results(file_path)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    return results

def parse_text_results(file_path):
    """
    Parse text-based result files (adjust based on your output format)
    """
    # This is a placeholder - adjust based on your actual output format
    # Common metrics in neuron segmentation
    metrics = {
        'rand_score': 0.0,
        'vi_split': 0.0,
        'vi_merge': 0.0,
        'adapted_rand': 0.0
    }
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Parse your specific format here
            # This is just an example
            lines = content.strip().split('\n')
            for line in lines:
                if 'rand_score' in line:
                    metrics['rand_score'] = float(line.split(':')[-1].strip())
                elif 'vi_split' in line:
                    metrics['vi_split'] = float(line.split(':')[-1].strip())
                # Add more parsing as needed
    except:
        pass
    
    return metrics

def extract_best_metrics(results):
    """
    Extract best metrics for each method
    """
    best_results = {}
    
    for method, method_results in results.items():
        if not method_results:
            continue
            
        # Find best results across all thresholds and merge functions
        best_metrics = {}
        for (merge_func, threshold), metrics in method_results.items():
            for metric_name, value in metrics.items():
                if metric_name not in best_metrics:
                    best_metrics[metric_name] = {'value': value, 'params': (merge_func, threshold)}
                else:
                    # For most metrics, higher is better (like rand_score, adapted_rand)
                    # For VI scores, lower is better
                    if metric_name in ['vi_split', 'vi_merge']:
                        if value < best_metrics[metric_name]['value']:
                            best_metrics[metric_name] = {'value': value, 'params': (merge_func, threshold)}
                    else:
                        if value > best_metrics[metric_name]['value']:
                            best_metrics[metric_name] = {'value': value, 'params': (merge_func, threshold)}
        
        best_results[method] = best_metrics
    
    return best_results

def plot_comparison(best_results, sample_name='B'):
    """
    Create bar plots comparing SOTA and CM-GLLF results
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Common metrics to plot
    metrics_to_plot = ['rand_score', 'adapted_rand', 'vi_split', 'vi_merge']
    
    # Find best CM-GLLF method
    cm_gllf_methods = {k: v for k, v in best_results.items() if k != 'SOTA'}
    best_cm_gllf_method = find_best_cm_gllf_method(cm_gllf_methods)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'SOTA vs CM-GLLF Comparison - Sample {sample_name}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Prepare data for plotting
        methods = []
        values = []
        colors = []
        
        # SOTA
        if 'SOTA' in best_results and metric in best_results['SOTA']:
            methods.append('SOTA')
            values.append(best_results['SOTA'][metric]['value'])
            colors.append('red')
        
        # Best CM-GLLF
        if best_cm_gllf_method and metric in best_results[best_cm_gllf_method]:
            methods.append(f'CM-GLLF\n{best_cm_gllf_method}')
            values.append(best_results[best_cm_gllf_method][metric]['value'])
            colors.append('blue')
        
        # Create bar plot
        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Customize the plot
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Adjust y-axis limits for better visualization
        if values:
            y_min, y_max = min(values), max(values)
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range*0.1, y_max + y_range*0.15)
    
    plt.tight_layout()
    plt.savefig(f'comparison_sample_{sample_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_cm_gllf_method

def find_best_cm_gllf_method(cm_gllf_methods):
    """
    Find the best CM-GLLF method based on overall performance
    """
    if not cm_gllf_methods:
        return None
    
    # Score each method based on multiple metrics
    method_scores = {}
    
    for method, metrics in cm_gllf_methods.items():
        score = 0
        count = 0
        
        for metric_name, metric_data in metrics.items():
            value = metric_data['value']
            
            # Normalize scores (higher is better for all)
            if metric_name in ['vi_split', 'vi_merge']:
                # For VI metrics, lower is better, so invert
                score += (1 - value) if value < 1 else 0
            else:
                # For other metrics, higher is better
                score += value
            count += 1
        
        if count > 0:
            method_scores[method] = score / count
    
    # Return method with highest average score
    if method_scores:
        return max(method_scores, key=method_scores.get)
    return None

def print_detailed_results(best_results, best_cm_gllf_method):
    """
    Print detailed comparison results
    """
    print("="*60)
    print("DETAILED COMPARISON RESULTS")
    print("="*60)
    
    if best_cm_gllf_method:
        print(f"Best CM-GLLF Method: {best_cm_gllf_method}")
        print("-"*40)
    
    for method in ['SOTA', best_cm_gllf_method]:
        if method and method in best_results:
            print(f"\n{method} Results:")
            for metric, data in best_results[method].items():
                params = data['params']
                print(f"  {metric}: {data['value']:.4f} (merge_func: {params[0]}, threshold: {params[1]})")
    
    print("\n" + "="*60)

def main():
    """
    Main function to run the comparison analysis
    """
    sample_name = 'B'
    seg_path = f'../data/segs/{sample_name}/whole'
    
    # Parse all results
    print("Parsing evaluation results...")
    results = parse_evaluation_results(seg_path, sample_name)
    
    if not results:
        print("No results found. Make sure the segmentation has been completed.")
        return
    
    # Extract best metrics for each method
    print("Extracting best metrics...")
    best_results = extract_best_metrics(results)
    
    # Find best CM-GLLF method
    cm_gllf_methods = {k: v for k, v in best_results.items() if k != 'SOTA'}
    best_cm_gllf_method = find_best_cm_gllf_method(cm_gllf_methods)
    
    # Print detailed results
    print_detailed_results(best_results, best_cm_gllf_method)
    
    # Create comparison plots
    print("Creating comparison plots...")
    plot_comparison(best_results, sample_name)
    
    print(f"\nComparison complete! Best CM-GLLF parameters: {best_cm_gllf_method}")

if __name__ == "__main__":
    main()