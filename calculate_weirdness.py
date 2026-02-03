#!/usr/bin/env python3
"""
Language Weirdness Calculator
Analyzes WALS dataset to compute weirdness scores based on feature rarity.
"""

import pandas as pd
import numpy as np
import json
from collections import Counter

def calculate_feature_rarity_scores(df, feature_cols):
    """
    For each feature, calculate rarity scores for each possible value.
    Rarity = 1 - (frequency of value / total non-missing values)
    Returns a dictionary mapping feature -> value -> rarity_score
    """
    rarity_scores = {}
    feature_stats = {}
    
    for feature in feature_cols:
        # Get non-missing values
        values = df[feature].dropna()
        
        if len(values) == 0:
            continue
            
        # Count frequency of each value
        value_counts = Counter(values)
        total = len(values)
        
        # Calculate rarity score for each value (0 = most common, 1 = rarest)
        feature_rarity = {}
        for value, count in value_counts.items():
            frequency = count / total
            rarity = 1 - frequency
            feature_rarity[value] = rarity
        
        rarity_scores[feature] = feature_rarity
        
        # Store stats for analysis
        feature_stats[feature] = {
            'total_responses': total,
            'unique_values': len(value_counts),
            'most_common': value_counts.most_common(1)[0] if value_counts else None
        }
    
    return rarity_scores, feature_stats


def calculate_weirdness_scores(df, feature_cols, rarity_scores):
    """
    Calculate weirdness score for each language.
    Weirdness = average rarity across all features that language has data for.
    """
    weirdness_data = []
    
    for idx, row in df.iterrows():
        language_name = row['Name']
        
        # Collect rarity scores for all features this language has
        language_rarities = []
        feature_contributions = []
        
        for feature in feature_cols:
            value = row[feature]
            
            # Skip if missing or if we don't have rarity scores
            if pd.isna(value) or feature not in rarity_scores:
                continue
            
            # Get rarity score for this value
            if value in rarity_scores[feature]:
                rarity = rarity_scores[feature][value]
                language_rarities.append(rarity)
                feature_contributions.append({
                    'feature': feature,
                    'value': value,
                    'rarity': rarity
                })
        
        # Calculate average weirdness
        if language_rarities:
            weirdness_score = np.mean(language_rarities)
            num_features = len(language_rarities)
        else:
            weirdness_score = None
            num_features = 0
        
        # Sort features by contribution to weirdness
        feature_contributions.sort(key=lambda x: x['rarity'], reverse=True)
        
        weirdness_data.append({
            'name': language_name,
            'wals_code': row['wals_code'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'family': row['family'],
            'genus': row['genus'],
            'macroarea': row['macroarea'],
            'weirdness_score': weirdness_score,
            'num_features': num_features,
            'top_weird_features': feature_contributions[:5]  # Top 5 weirdest features
        })
    
    return pd.DataFrame(weirdness_data)


def main():
    print("Loading WALS dataset...")
    df = pd.read_csv('language.csv')
    
    # Get feature columns (everything after 'countrycodes')
    feature_cols = df.columns[10:].tolist()
    print(f"Found {len(feature_cols)} feature columns")
    
    print("\nCalculating rarity scores for each feature value...")
    rarity_scores, feature_stats = calculate_feature_rarity_scores(df, feature_cols)
    
    print(f"Processed {len(rarity_scores)} features with data")
    
    # Show some examples
    print("\nExample: Feature '10A Vowel Nasalization'")
    if '10A Vowel Nasalization' in rarity_scores:
        for value, rarity in sorted(rarity_scores['10A Vowel Nasalization'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {value}: rarity = {rarity:.3f}")
    
    print("\nCalculating weirdness scores for each language...")
    weirdness_df = calculate_weirdness_scores(df, feature_cols, rarity_scores)
    
    # Remove languages with no weirdness score
    weirdness_df = weirdness_df[weirdness_df['weirdness_score'].notna()]
    
    print(f"\nCalculated scores for {len(weirdness_df)} languages")
    
    # Filter for languages with at least 10 features for robust scoring
    MIN_FEATURES = 30
    weirdness_df_robust = weirdness_df[weirdness_df['num_features'] >= MIN_FEATURES].copy()
    print(f"Languages with at least {MIN_FEATURES} features: {len(weirdness_df_robust)}")
    
    # Show top 10 weirdest languages
    print("\n" + "="*70)
    print("TOP 10 WEIRDEST LANGUAGES (with at least 10 features):")
    print("="*70)
    top_10 = weirdness_df_robust.nlargest(10, 'weirdness_score')
    for idx, row in top_10.iterrows():
        print(f"\n{row['name']} (Family: {row['family']})")
        print(f"  Weirdness Score: {row['weirdness_score']:.4f}")
        print(f"  Based on {row['num_features']} features")
        print(f"  Top weird features:")
        for feat in row['top_weird_features'][:3]:
            print(f"    - {feat['feature']}: {feat['value']} (rarity: {feat['rarity']:.3f})")
    
    # Show bottom 10 (most "normal" languages)
    print("\n" + "="*70)
    print("TOP 10 MOST 'NORMAL' LANGUAGES (with at least 10 features):")
    print("="*70)
    bottom_10 = weirdness_df_robust.nsmallest(10, 'weirdness_score')
    for idx, row in bottom_10.iterrows():
        print(f"{row['name']}: {row['weirdness_score']:.4f} ({row['num_features']} features, Family: {row['family']})")
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS (all languages):")
    print("="*70)
    print(f"Mean weirdness: {weirdness_df['weirdness_score'].mean():.4f}")
    print(f"Median weirdness: {weirdness_df['weirdness_score'].median():.4f}")
    print(f"Std deviation: {weirdness_df['weirdness_score'].std():.4f}")
    print(f"Min: {weirdness_df['weirdness_score'].min():.4f}")
    print(f"Max: {weirdness_df['weirdness_score'].max():.4f}")
    
    print("\n" + "="*70)
    print(f"STATISTICS (languages with at least {MIN_FEATURES} features):")
    print("="*70)
    print(f"Mean weirdness: {weirdness_df_robust['weirdness_score'].mean():.4f}")
    print(f"Median weirdness: {weirdness_df_robust['weirdness_score'].median():.4f}")
    print(f"Std deviation: {weirdness_df_robust['weirdness_score'].std():.4f}")
    print(f"Min: {weirdness_df_robust['weirdness_score'].min():.4f}")
    print(f"Max: {weirdness_df_robust['weirdness_score'].max():.4f}")
    
    # Save results
    print("\nSaving results...")
    
    # Save full dataset as CSV
    weirdness_df.to_csv('language_weirdness_scores.csv', index=False)
    print("  ✓ Saved language_weirdness_scores.csv")
    
    # Save JSON for web map (simplified version)
    map_data = []
    for idx, row in weirdness_df.iterrows():
        map_data.append({
            'name': row['name'],
            'lat': float(row['latitude']) if pd.notna(row['latitude']) else None,
            'lon': float(row['longitude']) if pd.notna(row['longitude']) else None,
            'family': row['family'] if pd.notna(row['family']) else 'Unknown',
            'genus': row['genus'] if pd.notna(row['genus']) else 'Unknown',
            'weirdness': float(row['weirdness_score']),
            'numFeatures': int(row['num_features']),
            'topFeatures': [
                {
                    'feature': f['feature'],
                    'value': f['value'],
                    'rarity': float(f['rarity'])
                }
                for f in row['top_weird_features'][:3]
            ]
        })
    
    # Filter out languages with invalid coordinates
    map_data = [lang for lang in map_data if lang['lat'] is not None and lang['lon'] is not None]
    
    with open('language_data.json', 'w') as f:
        json.dump(map_data, f, indent=2)
    print("  ✓ Saved language_data.json")
    
    # Save feature statistics
    with open('feature_stats.json', 'w') as f:
        json.dump(feature_stats, f, indent=2)
    print("  ✓ Saved feature_stats.json")
    
    print("\nDone! Ready to create the interactive map.")
    

if __name__ == '__main__':
    main()
