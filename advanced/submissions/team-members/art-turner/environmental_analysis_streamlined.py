import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def main():
    print("TETUAN CITY - ENVIRONMENTAL VARIABLES vs POWER CONSUMPTION ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv('data/Tetuan City power consumption.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%m/%d/%Y %H:%M')
    
    # Rename power consumption columns
    df['Zone1'] = df['Zone 1 Power Consumption']
    df['Zone2'] = df['Zone 2  Power Consumption']
    df['Zone3'] = df['Zone 3  Power Consumption']
    df['Total_Power'] = df['Zone1'] + df['Zone2'] + df['Zone3']
    
    # Environmental variables
    env_vars = ['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    power_vars = ['Zone1', 'Zone2', 'Zone3', 'Total_Power']
    
    print(f"Analyzing correlations between {len(env_vars)} environmental variables and {len(power_vars)} power measures")
    print(f"Dataset: {len(df):,} records")
    
    # Calculate correlations
    print("\\n" + "="*80)
    print("CORRELATION ANALYSIS RESULTS")
    print("="*80)
    
    correlations = {}
    
    for power_var in power_vars:
        correlations[power_var] = {}
        print(f"\\n{power_var} CORRELATIONS:")
        print("-" * 40)
        
        env_corrs = []
        for env_var in env_vars:
            corr_coef, p_value = pearsonr(df[env_var], df[power_var])
            correlations[power_var][env_var] = {
                'correlation': corr_coef,
                'p_value': p_value
            }
            env_corrs.append((env_var, corr_coef, p_value))
        
        # Sort by absolute correlation strength
        env_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for env_var, corr, p_val in env_corrs:
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            direction = "Positive" if corr > 0 else "Negative"
            strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
            print(f"  {env_var:25}: {corr:7.4f} {sig:3} ({direction}, {strength})")
    
    # Analyze strongest correlations overall
    print("\\n" + "="*80)
    print("STRONGEST ENVIRONMENTAL CORRELATIONS")
    print("="*80)
    
    # For total power
    total_corrs = []
    for env_var in env_vars:
        corr = correlations['Total_Power'][env_var]['correlation']
        total_corrs.append((env_var, corr))
    
    total_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("\\nRANKED BY TOTAL POWER CORRELATION STRENGTH:")
    for i, (env_var, corr) in enumerate(total_corrs, 1):
        direction = "Positively" if corr > 0 else "Negatively"
        strength = "Strongly" if abs(corr) > 0.5 else "Moderately" if abs(corr) > 0.3 else "Weakly"
        print(f"  {i}. {env_var:25}: {corr:7.4f} ({strength} {direction.lower()})")
    
    # Identify inverse correlations
    print("\\n" + "="*80)
    print("INVERSE CORRELATIONS ANALYSIS")
    print("="*80)
    
    inverse_found = False
    zones_only = ['Zone1', 'Zone2', 'Zone3']
    
    for zone in zones_only:
        zone_inverses = []
        for env_var in env_vars:
            corr = correlations[zone][env_var]['correlation']
            p_val = correlations[zone][env_var]['p_value']
            
            if corr < -0.05 and p_val < 0.05:  # Any negative correlation
                zone_inverses.append((env_var, corr, p_val))
        
        if zone_inverses:
            inverse_found = True
            zone_inverses.sort(key=lambda x: x[1])  # Sort by most negative
            print(f"\\n{zone} INVERSE CORRELATIONS:")
            for env_var, corr, p_val in zone_inverses:
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
                print(f"  {env_var:25}: {corr:7.4f} {sig} ({strength})")
    
    if not inverse_found:
        print("\\nNo significant inverse correlations found in any zones.")
    
    # Zone comparison analysis
    print("\\n" + "="*80)
    print("ZONE COMPARISON ANALYSIS")
    print("="*80)
    
    print("\\nCORRELATION DIFFERENCES ACROSS ZONES:")
    zone_variations = {}
    
    for env_var in env_vars:
        print(f"\\n{env_var}:")
        zone_corrs = []
        
        for zone in zones_only:
            corr = correlations[zone][env_var]['correlation']
            zone_corrs.append((zone, corr))
            print(f"  {zone}: {corr:7.4f}")
        
        # Calculate variation across zones
        corr_values = [corr for _, corr in zone_corrs]
        min_corr = min(corr_values)
        max_corr = max(corr_values)
        variation = max_corr - min_corr
        zone_variations[env_var] = variation
        
        print(f"  Range: {min_corr:.4f} to {max_corr:.4f}")
        print(f"  Variation: {variation:.4f}", end="")
        
        if variation > 0.3:
            print(" - LARGE DIFFERENCE!")
        elif variation > 0.15:
            print(" - Moderate difference")
        else:
            print(" - Similar across zones")
    
    # Answer the three questions
    print("\\n" + "="*80)
    print("ANSWERS TO THE THREE KEY QUESTIONS")
    print("="*80)
    
    print("\\nQ: Which environmental variables correlate most with energy usage?")
    print("A:")
    print("   CORRELATION STRENGTH RANKING (Total Power):")
    
    for i, (env_var, corr) in enumerate(total_corrs, 1):
        abs_corr = abs(corr)
        if abs_corr > 0.5:
            strength_desc = "STRONG correlation"
        elif abs_corr > 0.3:
            strength_desc = "MODERATE correlation"
        elif abs_corr > 0.1:
            strength_desc = "WEAK correlation"
        else:
            strength_desc = "VERY WEAK correlation"
        
        direction = "positive" if corr > 0 else "negative"
        print(f"   {i}. {env_var}: {corr:.4f} ({strength_desc}, {direction})")
    
    strongest_var, strongest_corr = total_corrs[0]
    print(f"\\n   {strongest_var} shows the strongest relationship with power consumption.")
    
    if strongest_corr > 0:
        print(f"   Higher {strongest_var.lower()} is associated with higher energy usage.")
    else:
        print(f"   Higher {strongest_var.lower()} is associated with lower energy usage.")
    
    print("\\n\\nQ: Are any variables inversely correlated with demand in specific zones?")
    print("A:")
    
    if inverse_found:
        print("   Yes, several inverse correlations were found:")
        for zone in zones_only:
            zone_inverses = []
            for env_var in env_vars:
                corr = correlations[zone][env_var]['correlation']
                p_val = correlations[zone][env_var]['p_value']
                if corr < -0.05 and p_val < 0.05:
                    zone_inverses.append((env_var, corr))
            
            if zone_inverses:
                zone_inverses.sort(key=lambda x: x[1])
                print(f"   {zone}:")
                for env_var, corr in zone_inverses:
                    print(f"     {env_var}: {corr:.4f} (as {env_var.lower()} increases, power decreases)")
    else:
        print("   No significant inverse correlations found.")
        print("   All environmental variables show positive or negligible correlations.")
    
    print("\\n\\nQ: Did your analysis differ across zones? Why might that be?")
    print("A:")
    print("   ZONE DIFFERENCES:")
    
    # Sort variations by magnitude
    sorted_variations = sorted(zone_variations.items(), key=lambda x: x[1], reverse=True)
    
    significant_differences = False
    for env_var, variation in sorted_variations:
        if variation > 0.2:
            significant_differences = True
            print(f"   {env_var}: SIGNIFICANT differences across zones (range: {variation:.4f})")
        elif variation > 0.1:
            print(f"   {env_var}: Moderate differences across zones (range: {variation:.4f})")
        else:
            print(f"   {env_var}: Similar responses across zones (range: {variation:.4f})")
    
    if significant_differences:
        print("\\n   REASONS FOR ZONE DIFFERENCES:")
        print("   • BUILDING TYPES: Residential vs Commercial vs Industrial usage patterns")
        print("   • COOLING/HEATING SYSTEMS: Different HVAC efficiency and technology")
        print("   • OCCUPANCY PATTERNS: Varying schedules and density across zones")
        print("   • BUILDING CHARACTERISTICS: Age, insulation, orientation, size")
        print("   • GEOGRAPHIC FACTORS: Urban heat island effects, elevation, exposure")
        print("   • ECONOMIC FACTORS: Income levels affecting conservation behavior")
        print("   • INFRASTRUCTURE: Grid efficiency, voltage regulation differences")
    else:
        print("\\n   Zones show remarkably similar responses to environmental conditions.")
        print("   This suggests consistent building types and usage patterns across Tetuan.")
    
    print("\\n" + "="*80)
    print("ENVIRONMENTAL CORRELATION ANALYSIS COMPLETED")
    print("="*80)
    
    return correlations

if __name__ == "__main__":
    results = main()