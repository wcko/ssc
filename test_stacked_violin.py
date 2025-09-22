#!/usr/bin/env python3
"""
Test script for stacked_vlnplot function
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
import scanpy as sc

# Create mock AnnData object for testing
def create_mock_adata():
    """Create a mock AnnData object with some realistic-looking data"""
    np.random.seed(42)

    # Create expression data
    n_cells = 500
    n_genes = 20

    # Generate some expression data
    X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(float)

    # Create gene names (mix of real and mock genes)
    genes = ['CD3D', 'CD8A', 'CD4', 'CD19', 'MS4A1', 'CD14', 'FCGR3A', 'CD68',
            'NKG7', 'GNLY', 'IL7R', 'CCR7', 'S100A8', 'S100A9', 'LYZ', 'CST3',
            'FCER1A', 'CD1C', 'PPBP', 'PF4']

    # Create cell types
    cell_types = np.random.choice(['T_cells', 'B_cells', 'Monocytes', 'NK_cells', 'Dendritic'],
                                 size=n_cells, p=[0.4, 0.2, 0.2, 0.1, 0.1])

    # Make expression more realistic by boosting certain genes in certain cell types
    for i, cell_type in enumerate(['T_cells', 'B_cells', 'Monocytes', 'NK_cells', 'Dendritic']):
        mask = cell_types == cell_type
        if cell_type == 'T_cells':
            # Boost T cell markers
            X[mask, 0:3] *= 3  # CD3D, CD8A, CD4
        elif cell_type == 'B_cells':
            # Boost B cell markers
            X[mask, 3:5] *= 4  # CD19, MS4A1
        elif cell_type == 'Monocytes':
            # Boost monocyte markers
            X[mask, 5:8] *= 3  # CD14, FCGR3A, CD68
        elif cell_type == 'NK_cells':
            # Boost NK markers
            X[mask, 8:10] *= 4  # NKG7, GNLY

    # Create AnnData object
    import anndata as ad
    adata = ad.AnnData(X)
    adata.var_names = genes
    adata.obs['cell_type'] = cell_types
    adata.obs_names = [f'cell_{i}' for i in range(n_cells)]

    return adata

def test_basic_functionality():
    """Test basic stacked violin plot functionality"""
    print("Creating mock data...")
    adata = create_mock_adata()

    print("Testing basic stacked_vlnplot...")
    from ssc.plotting import stacked_vlnplot

    # Test basic plot
    genes_to_plot = ['CD3D', 'CD8A', 'CD19', 'MS4A1', 'CD14', 'NKG7']

    fig = stacked_vlnplot(
        adata,
        genes=genes_to_plot,
        group_by='cell_type',
        title='Test Stacked Violin Plot'
    )

    plt.savefig('/Users/weicheko/dropbox_umass/Projects/ssc/test_basic_stacked_violin.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Basic plot saved as test_basic_stacked_violin.png")

def test_with_marker_groups():
    """Test stacked violin plot with marker groups"""
    print("Testing with marker groups...")
    adata = create_mock_adata()

    from ssc.plotting import stacked_vlnplot

    # Define marker groups
    marker_groups = {
        'T_cell_markers': ['CD3D', 'CD8A', 'CD4'],
        'B_cell_markers': ['CD19', 'MS4A1'],
        'Myeloid_markers': ['CD14', 'CD68'],
        'NK_markers': ['NKG7', 'GNLY']
    }

    genes_to_plot = ['CD3D', 'CD8A', 'CD4', 'CD19', 'MS4A1', 'CD14', 'CD68', 'NKG7', 'GNLY']

    fig = stacked_vlnplot(
        adata,
        genes=genes_to_plot,
        group_by='cell_type',
        marker_groups=marker_groups,
        show_gene_groups=True,
        title='Stacked Violin Plot with Marker Groups'
    )

    plt.savefig('/Users/weicheko/dropbox_umass/Projects/ssc/test_marker_groups_stacked_violin.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Marker groups plot saved as test_marker_groups_stacked_violin.png")

def test_inverted_axes():
    """Test inverted axes functionality"""
    print("Testing inverted axes...")
    adata = create_mock_adata()

    from ssc.plotting import stacked_vlnplot

    genes_to_plot = ['CD3D', 'CD8A', 'CD19', 'MS4A1', 'CD14', 'NKG7']

    fig = stacked_vlnplot(
        adata,
        genes=genes_to_plot,
        group_by='cell_type',
        invert_axes=True,
        title='Inverted Axes - Groups on X-axis'
    )

    plt.savefig('/Users/weicheko/dropbox_umass/Projects/ssc/test_inverted_stacked_violin.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Inverted axes plot saved as test_inverted_stacked_violin.png")

if __name__ == "__main__":
    print("Testing stacked_vlnplot functionality...")

    try:
        test_basic_functionality()
        test_with_marker_groups()
        test_inverted_axes()
        print("\n✅ All tests passed successfully!")
        print("Check the generated PNG files to verify the plots look correct.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()