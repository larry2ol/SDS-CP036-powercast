"""
Complete Model Analysis Pipeline
Runs interpretability analysis, error pattern analysis, and generates comprehensive report
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse
import sys

from interpretability_analysis import run_interpretability_analysis
from error_pattern_analysis import run_comprehensive_error_analysis


def create_analysis_report(interpretability_results, error_results, model_path, output_dir):
    """Create a comprehensive analysis report"""
    
    report_path = Path(output_dir) / 'comprehensive_analysis_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Power Forecasting Model Analysis Report\n\n")
        
        f.write(f"**Model Analyzed**: `{model_path}`\n")
        f.write(f"**Model Type**: {interpretability_results.get('model_type', 'Unknown')}\n")
        f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report provides comprehensive interpretability and error analysis for the power forecasting model.\n\n")
        
        # Model Performance Summary
        f.write("## Model Performance Summary\n\n")
        if 'error_analysis' in interpretability_results:
            error_analysis = interpretability_results['error_analysis']
            f.write("| Zone | RMSE | R² | Bias (Mean Residual) |\n")
            f.write("|------|------|----|--------------------|n")\
            
            for zone_name, stats in error_analysis.items():
                rmse = stats.get('rmse', 0)
                r2 = stats.get('r2', 0)
                bias = error_results['distribution_analysis'][zone_name]['mean'] if zone_name in error_results['distribution_analysis'] else 0
                f.write(f"| {zone_name} | {rmse:.1f} | {r2:.3f} | {bias:.2f} |\n")
        
        f.write("\n")
        
        # Feature Importance Analysis
        f.write("## Feature Importance Analysis\n\n")
        if 'feature_importance' in interpretability_results:
            feature_importance = interpretability_results['feature_importance']
            
            for zone_name, importance_dict in feature_importance.items():
                f.write(f"### {zone_name}\n\n")
                
                # Sort features by importance
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                f.write("| Feature | Importance Score |\n")
                f.write("|---------|------------------|\n")
                
                for feature, score in sorted_features[:10]:  # Top 10
                    f.write(f"| {feature} | {score:.4f} |\n")
                f.write("\n")
        
        # Temporal Analysis
        f.write("## Temporal Importance Analysis\n\n")
        if 'temporal_importance' in interpretability_results:
            temporal_importance = interpretability_results['temporal_importance']
            
            for zone_name, importance_array in temporal_importance.items():
                most_important_timestep = np.argmax(importance_array)
                max_importance = np.max(importance_array)
                
                f.write(f"**{zone_name}**:\n")
                f.write(f"- Most critical timestep: T-{most_important_timestep} (importance: {max_importance:.4f})\n")
                f.write(f"- Shows dependency on information from {36 - most_important_timestep} steps ago\n\n")
        
        # Error Distribution Analysis
        f.write("## Error Distribution Analysis\n\n")
        if 'distribution_analysis' in error_results:
            dist_analysis = error_results['distribution_analysis']
            
            f.write("| Zone | Mean Residual | Std Residual | Skewness | Outliers % | Normal? |\n")
            f.write("|------|---------------|--------------|----------|------------|--------|\n")
            
            for zone_name, stats in dist_analysis.items():
                mean_res = stats['mean']
                std_res = stats['std']
                skewness = stats['skewness']
                outlier_pct = stats['outlier_percentage']
                is_normal = "Yes" if stats['shapiro_p_value'] >= 0.05 else "No"
                
                f.write(f"| {zone_name} | {mean_res:.3f} | {std_res:.1f} | {skewness:.3f} | {outlier_pct:.1f}% | {is_normal} |\n")
        
        f.write("\n")
        
        # Key Insights
        f.write("## Key Insights\n\n")
        
        # Bias Analysis
        f.write("### Bias Analysis\n")
        if 'distribution_analysis' in error_results:
            for zone_name, stats in error_results['distribution_analysis'].items():
                bias = stats['mean']
                if abs(bias) > 100:  # Significant bias
                    direction = "over-predicting" if bias < 0 else "under-predicting"
                    f.write(f"- **{zone_name}**: Model is {direction} by {abs(bias):.1f} units on average\n")
                else:
                    f.write(f"- **{zone_name}**: Low bias ({bias:.2f} units)\n")
        f.write("\n")
        
        # Distribution Issues
        f.write("### Distribution Issues\n")
        if 'distribution_analysis' in error_results:
            for zone_name, stats in error_results['distribution_analysis'].items():
                if stats['shapiro_p_value'] < 0.05:
                    f.write(f"- **{zone_name}**: Residuals are not normally distributed (p={stats['shapiro_p_value']:.4f})\n")
                if abs(stats['skewness']) > 0.5:
                    skew_direction = "right" if stats['skewness'] > 0 else "left"
                    f.write(f"- **{zone_name}**: Residuals are {skew_direction}-skewed ({stats['skewness']:.3f})\n")
                if stats['outlier_percentage'] > 5:
                    f.write(f"- **{zone_name}**: High outlier rate ({stats['outlier_percentage']:.1f}%)\n")
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        f.write("### Model Improvements\n")
        
        # Check if any zone has poor performance
        if 'error_analysis' in interpretability_results:
            poor_zones = [zone for zone, stats in interpretability_results['error_analysis'].items() 
                         if stats.get('r2', 0) < 0.5]
            if poor_zones:
                f.write(f"- Focus on improving predictions for: {', '.join(poor_zones)}\n")
        
        # Check for bias issues
        if 'distribution_analysis' in error_results:
            biased_zones = [zone for zone, stats in error_results['distribution_analysis'].items() 
                           if abs(stats['mean']) > 100]
            if biased_zones:
                f.write(f"- Address systematic bias in: {', '.join(biased_zones)}\n")
        
        f.write("- Consider ensemble methods to reduce variance\n")
        f.write("- Investigate outliers for data quality issues\n")
        f.write("- Consider robust loss functions if heavy-tailed residuals persist\n\n")
        
        f.write("### Feature Engineering\n")
        if 'feature_importance' in interpretability_results:
            # Find consistently important features across zones
            all_features = set()
            for zone_importance in interpretability_results['feature_importance'].values():
                all_features.update(zone_importance.keys())
            
            feature_avg_importance = {}
            for feature in all_features:
                importances = [zone_importance.get(feature, 0) 
                             for zone_importance in interpretability_results['feature_importance'].values()]
                feature_avg_importance[feature] = np.mean(importances)
            
            top_features = sorted(feature_avg_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            f.write("- Most important features across zones:\n")
            for feature, importance in top_features:
                f.write(f"  - {feature} (avg importance: {importance:.4f})\n")
        
        f.write("\n")
        
        f.write("## Files Generated\n\n")
        f.write("### Interpretability Analysis\n")
        f.write("- `interpretability_results/temporal_importance.png`\n")
        f.write("- `interpretability_results/feature_importance.png`\n")
        f.write("- `interpretability_results/saliency_heatmap.png`\n")
        f.write("- `interpretability_results/interpretability_results.json`\n\n")
        
        f.write("### Error Analysis\n")
        f.write("- `error_analysis_results/temporal_error_patterns.png`\n")
        f.write("- `error_analysis_results/magnitude_error_analysis.png`\n")
        f.write("- `error_analysis_results/residual_distributions.png`\n")
        f.write("- `error_analysis_results/comprehensive_error_analysis.json`\n\n")
        
        f.write("---\n")
        f.write("*Report generated by comprehensive analysis pipeline*\n")
    
    print(f"Comprehensive report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive model analysis')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file (.pth)')
    parser.add_argument('--output_dir', type=str, default='full_analysis_results',
                       help='Output directory for all results')
    parser.add_argument('--skip_interpretability', action='store_true',
                       help='Skip interpretability analysis (faster)')
    parser.add_argument('--skip_error_analysis', action='store_true',
                       help='Skip error pattern analysis')
    
    args = parser.parse_args()
    
    # Validate model path
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create main output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("COMPREHENSIVE MODEL ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print()
    
    results = {}
    
    # Run interpretability analysis
    if not args.skip_interpretability:
        print("Running interpretability analysis...")
        interp_dir = output_path / 'interpretability_results'
        try:
            interpretability_results = run_interpretability_analysis(
                args.model_path, 
                data_path=None,  # Uses default data files
                output_dir=str(interp_dir)
            )
            results['interpretability'] = interpretability_results
            print("✓ Interpretability analysis completed")
        except Exception as e:
            print(f"✗ Interpretability analysis failed: {e}")
            results['interpretability'] = {}
    else:
        print("Skipping interpretability analysis...")
        results['interpretability'] = {}
    
    print()
    
    # Run error pattern analysis
    if not args.skip_error_analysis:
        print("Running error pattern analysis...")
        error_dir = output_path / 'error_analysis_results'
        try:
            error_results = run_comprehensive_error_analysis(
                args.model_path,
                output_dir=str(error_dir)
            )
            results['error_analysis'] = error_results
            print("✓ Error analysis completed")
        except Exception as e:
            print(f"✗ Error analysis failed: {e}")
            results['error_analysis'] = {}
    else:
        print("Skipping error analysis...")
        results['error_analysis'] = {}
    
    print()
    
    # Generate comprehensive report
    print("Generating comprehensive report...")
    try:
        # Import pandas for report generation
        import pandas as pd
        
        report_path = create_analysis_report(
            results.get('interpretability', {}),
            results.get('error_analysis', {}),
            args.model_path,
            args.output_dir
        )
        print("✓ Comprehensive report generated")
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
    
    print()
    print("=" * 60)
    print("ANALYSIS PIPELINE COMPLETE")
    print("=" * 60)
    print(f"All results saved to: {args.output_dir}")
    
    # Print quick summary
    if results.get('interpretability', {}).get('error_analysis'):
        print("\nQuick Performance Summary:")
        error_analysis = results['interpretability']['error_analysis']
        for zone_name, stats in error_analysis.items():
            rmse = stats.get('rmse', 0)
            r2 = stats.get('r2', 0)
            print(f"  {zone_name}: RMSE={rmse:.1f}, R²={r2:.3f}")


if __name__ == "__main__":
    main()