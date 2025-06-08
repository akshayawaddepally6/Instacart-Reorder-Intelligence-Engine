"""
Data Processing - FIXED VERSION
Simple, working version that handles data types correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("="*60)
    print("INSTACART DATA PROCESSING - FIXED VERSION")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    orders = pd.read_csv('data/order_products__train.csv')
    products = pd.read_csv('data/products.csv')
    aisles = pd.read_csv('data/aisles.csv')
    departments = pd.read_csv('data/departments.csv')
    print(f"✓ Loaded {len(orders):,} orders")
    
    # Merge to get full data
    print("\n2. Merging data...")
    products_full = products.merge(aisles, on='aisle_id').merge(departments, on='department_id')
    data = orders.merge(products_full, on='product_id')
    print(f"✓ Merged data: {len(data):,} rows")
    
    # Create product features
    print("\n3. Creating product features...")
    product_stats = orders.groupby('product_id').agg({
        'order_id': 'count',
        'reordered': ['sum', 'mean'],
        'add_to_cart_order': ['mean', 'median', 'std']
    })
    product_stats.columns = [
        'product_orders', 'product_reorders', 'product_reorder_rate',
        'avg_cart_position', 'median_cart_position', 'std_cart_position'
    ]
    product_stats['std_cart_position'] = product_stats['std_cart_position'].fillna(0)
    product_stats['product_popularity_rank'] = product_stats['product_orders'].rank(ascending=False)
    product_stats = product_stats.reset_index()
    
    data = data.merge(product_stats, on='product_id')
    print(f"✓ Added product features")
    
    # Create order features
    print("\n4. Creating order features...")
    order_stats = orders.groupby('order_id').agg({
        'product_id': 'count',
        'reordered': ['sum', 'mean']
    })
    order_stats.columns = ['order_size', 'order_reorders', 'order_reorder_rate']
    order_stats = order_stats.reset_index()
    
    data = data.merge(order_stats, on='order_id')
    print(f"✓ Added order features")
    
    # Create categorical features
    print("\n5. Creating categorical features...")
    dept_freq = data['department'].value_counts().to_dict()
    aisle_freq = data['aisle'].value_counts().to_dict()
    data['department_frequency'] = data['department'].map(dept_freq)
    data['aisle_frequency'] = data['aisle'].map(aisle_freq)
    print(f"✓ Added categorical features")
    
    # Create interaction features
    print("\n6. Creating interaction features...")
    data['reorder_x_position'] = data['product_reorder_rate'] * data['add_to_cart_order']
    data['popularity_score'] = data['product_orders'] / data['product_orders'].max()
    data['order_complexity'] = data['order_size'] * data['order_reorder_rate']
    print(f"✓ Added interaction features")
    
    # Select ONLY numeric features for ML
    print("\n7. Preparing final dataset...")
    feature_cols = [
        'product_orders', 'product_reorders', 'product_reorder_rate',
        'avg_cart_position', 'median_cart_position', 'std_cart_position',
        'product_popularity_rank', 'order_size', 'order_reorders',
        'order_reorder_rate', 'add_to_cart_order', 'department_frequency',
        'aisle_frequency', 'department_id', 'aisle_id',
        'reorder_x_position', 'popularity_score', 'order_complexity'
    ]
    
    # CRITICAL: Only numeric columns + target
    final_data = data[feature_cols + ['reordered']].copy()
    final_data = final_data.dropna()
    
    # Verify all numeric
    print(f"✓ Final dataset: {len(final_data):,} rows, {len(feature_cols)} features")
    print(f"✓ Reorder rate: {final_data['reordered'].mean():.2%}")
    
    # Check data types
    non_numeric = []
    for col in feature_cols:
        if final_data[col].dtype not in ['int64', 'float64']:
            non_numeric.append(col)
    
    if non_numeric:
        print(f"⚠ WARNING: Non-numeric columns found: {non_numeric}")
        print("Converting to numeric...")
        for col in non_numeric:
            final_data[col] = pd.to_numeric(final_data[col], errors='coerce')
        final_data = final_data.dropna()
        print(f"✓ After conversion: {len(final_data):,} rows")
    else:
        print("✓ All columns are numeric!")
    
    # Save
    print("\n8. Saving processed data...")
    Path('data/processed').mkdir(exist_ok=True)
    final_data.to_csv('data/processed/processed_data.csv', index=False)
    print(f"✓ Saved to data/processed/processed_data.csv")
    
    print("\n" + "="*60)
    print("✓ PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nDataset shape: {final_data.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Samples: {len(final_data):,}")
    print(f"Reorder rate: {final_data['reordered'].mean():.2%}")
    
    return final_data

if __name__ == "__main__":
    main()
