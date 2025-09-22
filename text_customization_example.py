"""
Example showing all text customization options for stacked violin plots
"""

# For your notebook - all text customization options:

print("🎨 Text Customization Options for Stacked Violin Plot")

# Example with various text size and rotation options
from ssc.plotting import stacked_vlnplot
import matplotlib.pyplot as plt

# Your marker set
markers = ['CD3D', 'TRAC', 'CD79A', 'MS4A1', 'CLEC10A', 'KRT14', 'GNLY', 'COL1A1']
marker_groups = {
    'T_cell': ['CD3D', 'TRAC'],
    'B_cell': ['CD79A', 'MS4A1'],
    'APC': ['CLEC10A'],
    'Epithelial': ['KRT14'],
    'Cytotoxic': ['GNLY'],
    'Stromal': ['COL1A1']
}

# Example 1: Default text sizes
fig1 = stacked_vlnplot(
    adata,
    genes=markers,
    group_by='cell_type_coarse',
    marker_groups=marker_groups,
    title='Default Text Sizes',
    # Default values shown for reference:
    title_fontsize=14,           # Main title
    gene_label_fontsize=10,      # Gene names (columns)
    group_label_fontsize=8,      # Cell type names (rows)
    gene_label_rotation=0,       # No rotation
    figsize=(12, 8)
)
plt.show()

# Example 2: Larger text with rotated gene labels
fig2 = stacked_vlnplot(
    adata,
    genes=markers,
    group_by='cell_type_coarse',
    marker_groups=marker_groups,
    title='Larger Text with Rotated Gene Labels',
    title_fontsize=18,           # Bigger title
    gene_label_fontsize=12,      # Bigger gene labels
    group_label_fontsize=10,     # Bigger group labels
    gene_label_rotation=45,      # Rotate gene names 45°
    figsize=(12, 8)
)
plt.show()

# Example 3: Publication style - clean and readable
fig3 = stacked_vlnplot(
    adata,
    genes=markers,
    group_by='cell_type_coarse',
    marker_groups=marker_groups,
    title='Publication Style - Clean & Readable',
    title_fontsize=16,           # Medium title
    gene_label_fontsize=11,      # Readable gene labels
    group_label_fontsize=9,      # Readable group labels
    gene_label_rotation=90,      # Vertical gene names (saves space)
    figsize=(10, 8),
    violin_alpha=0.8
)
plt.show()

print("📝 Text Customization Parameters:")
print()
print("title_fontsize:")
print("  • Controls main plot title size")
print("  • Default: 14, Range: 10-24 typically")
print()
print("gene_label_fontsize:")
print("  • Controls gene names (column headers)")
print("  • Default: 10, Range: 8-14 typically")
print()
print("group_label_fontsize:")
print("  • Controls cell type names (row labels)")
print("  • Default: 8, Range: 6-12 typically")
print()
print("gene_label_rotation:")
print("  • Rotates gene names (degrees)")
print("  • 0 = horizontal, 45 = diagonal, 90 = vertical")
print("  • Useful for long gene names or many genes")
print()
print("💡 Tips:")
print("  • For many genes: use gene_label_rotation=90")
print("  • For publication: increase all font sizes by 2-4")
print("  • For presentations: use larger fonts (16-20 for title)")
print("  • Long gene names: try 45° rotation for readability")