#!/usr/bin/env python3
"""
Basic usage examples for ssc violin plots.
"""

import ssc
import scanpy as sc

def main():
    # Load your data
    adata = sc.read_h5ad('your_data.h5ad')
    
    print("🎻 ssc Violin Plot Examples")
    print("=" * 40)
    
    # Example 1: Basic violin plot
    print("\n1. Basic violin plot")
    fig = ssc.vlnplot(adata, 'GNLY', 'cell_type')
    fig.savefig('example_basic.png', dpi=300, bbox_inches='tight')
    print("   Saved: example_basic.png")
    
    # Example 2: Split violin with custom colors
    print("\n2. Split violin with custom colors")
    fig = ssc.vlnplot(adata, 'GNLY', 'condition', 
                      split_by='treatment',
                      split_colors={'pre': 'lightblue', 'dupi': 'darkblue'})
    fig.savefig('example_split.png', dpi=300, bbox_inches='tight')
    print("   Saved: example_split.png")
    
    # Example 3: Faceted plot
    print("\n3. Faceted plot with custom ordering")
    fig = ssc.vlnplot(adata, 'GNLY', 'condition', 
                      facet_by='subject', facet_col='cell_type',
                      group_order=['Nonlesional', 'SADBE', 'Metal'],
                      show_legend=False)  # Clean look for faceted plots
    fig.savefig('example_faceted.png', dpi=300, bbox_inches='tight')
    print("   Saved: example_faceted.png")
    
    # Example 4: Publication-ready styling
    print("\n4. Publication-ready styling")
    fig = ssc.vlnplot(adata, 'GNLY', 'condition',
                      split_by='treatment',
                      title='GNLY Expression Analysis',
                      group_order=['Nonlesional', 'SADBE', 'Metal'],
                      split_colors={'pre': '#2E86AB', 'dupi': '#A23B72'},
                      figsize=(10, 6),
                      title_fontsize=16,
                      xlabel_fontsize=12,
                      ylabel_fontsize=12,
                      number_fontsize=8,
                      jitter_dot_size=8)
    fig.savefig('example_publication.png', dpi=300, bbox_inches='tight')
    print("   Saved: example_publication.png")

    # Grid layout for facets instead of horizontal layout
    fig = ssc.vlnplot(adata, 'IL4', 'condition',
                      facet_by='subject', facet_ncols=2,
                      facet_figsize=(12, 10))  # 2×2 subject grid

    # Complex faceting with grid control
    fig = ssc.vlnplot(adata, 'IL13', 'condition',
                      facet_by='subject', facet_col='tissue',
                      acet_ncols=2, facet_figsize=(12, 16))  # 2×2 per tissue, stacked
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    main()