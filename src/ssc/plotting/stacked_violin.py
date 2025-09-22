"""
Stacked violin plot functions for single-cell RNA sequencing data visualization.

This module provides compact violin plots arranged in a grid layout, similar to dotplots
but showing full expression distributions. Particularly useful for marker validation
and cell annotation justification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
from typing import Optional, Dict, List, Union, Tuple


def _reshape_expression_data(adata, genes: List[str], group_by: str, layer: Optional[str] = None) -> pd.DataFrame:
    """
    Reshape expression data from wide format (genes as columns) to long format
    (single 'gene' and 'expression' columns) for easier plotting.

    Similar to R's pivot_longer functionality.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    genes : List[str]
        List of gene names to extract
    group_by : str
        Column name in adata.obs for grouping
    layer : Optional[str]
        Layer to use for expression values. If None, uses .X

    Returns
    -------
    pd.DataFrame
        Long format dataframe with columns: gene, expression, group_by
    """
    # Get expression data
    if layer is None:
        if hasattr(adata.X, 'toarray'):
            expr_data = adata.X.toarray()
        else:
            expr_data = adata.X
    else:
        if hasattr(adata.layers[layer], 'toarray'):
            expr_data = adata.layers[layer].toarray()
        else:
            expr_data = adata.layers[layer]

    # Find gene indices
    gene_indices = []
    found_genes = []
    missing_genes = []

    for gene in genes:
        if gene in adata.var_names:
            gene_indices.append(list(adata.var_names).index(gene))
            found_genes.append(gene)
        else:
            missing_genes.append(gene)

    if missing_genes:
        warnings.warn(f"Genes not found in adata: {missing_genes}")

    if not found_genes:
        raise ValueError("No valid genes found in adata")

    # Extract expression data for found genes
    expr_subset = expr_data[:, gene_indices]

    # Create dataframe
    df_list = []
    for i, gene in enumerate(found_genes):
        gene_df = pd.DataFrame({
            'gene': gene,
            'expression': expr_subset[:, i],
            group_by: adata.obs[group_by].values,
            'cell_id': adata.obs_names
        })
        df_list.append(gene_df)

    # Combine all genes
    df = pd.concat(df_list, ignore_index=True)

    # Set gene as categorical to preserve order
    df['gene'] = pd.Categorical(df['gene'], categories=found_genes, ordered=True)

    return df


def _get_colors_from_dict(color_dict: Optional[Dict], categories: List, default_palette: str = 'tab10') -> List:
    """
    Get colors for categories, using provided dict or default palette.
    Adapted from violin.py pattern.
    """
    if color_dict is None:
        # Use seaborn palette
        colors = sns.color_palette(default_palette, len(categories))
    else:
        colors = []
        default_colors = sns.color_palette(default_palette, len(categories))
        for i, cat in enumerate(categories):
            if cat in color_dict:
                colors.append(color_dict[cat])
            else:
                colors.append(default_colors[i])

    return colors


def _create_stacked_grid(n_genes: int, n_groups: int,
                        figsize: Union[str, Tuple[float, float]] = 'auto',
                        invert_axes: bool = False) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create matplotlib subplot grid for stacked violin plots.

    Parameters
    ----------
    n_genes : int
        Number of genes (columns in default orientation)
    n_groups : int
        Number of groups (rows in default orientation)
    figsize : Union[str, Tuple[float, float]]
        Figure size. If 'auto', calculate based on grid size
    invert_axes : bool
        If True, swap genes and groups positioning

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure and axes array
    """
    if invert_axes:
        nrows, ncols = n_genes, n_groups
        if figsize == 'auto':
            figsize = (max(2 * n_groups, 8), max(1.5 * n_genes, 6))
    else:
        nrows, ncols = n_groups, n_genes
        if figsize == 'auto':
            figsize = (max(1.5 * n_genes, 8), max(2 * n_groups, 6))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                            constrained_layout=False)  # Disable to manually control layout

    # Ensure axes is always 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Adjust subplot spacing to leave room for annotation bar at bottom
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.1, right=0.95,
                       hspace=0.3, wspace=0.2)

    return fig, axes


def _plot_grid_violin(ax, data: np.ndarray, horizontal: bool = True,
                     color: str = 'lightblue', alpha: float = 0.7,
                     violin_width: float = 0.8, shared_range: Optional[Dict] = None) -> None:
    """
    Plot a single violin in a grid cell.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : np.ndarray
        Expression values for this gene-group combination
    horizontal : bool
        If True, plot horizontal violin (expression on x-axis)
    color : str
        Fill color for violin
    alpha : float
        Transparency level
    violin_width : float
        Width of violin relative to available space
    """
    if len(data) == 0:
        # No data - leave empty
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    # Remove any NaN values
    data = data[~np.isnan(data)]

    if len(data) == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    # For very small sample sizes, adjust violin width to avoid misleading shapes
    if len(data) < 10:
        violin_width = min(violin_width, 0.3)

    try:
        if horizontal:
            # Horizontal violin (expression on x-axis)
            # Use matplotlib's violinplot with vert=False for true horizontal violins
            parts = ax.violinplot([data], positions=[0.5], widths=[violin_width],
                                 vert=False, showmeans=False, showmedians=False, showextrema=False)

            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(alpha)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.5)

            ax.set_ylim(0, 1)
            # Set x-axis limits using shared range if provided, otherwise use data range
            if shared_range is not None:
                # Use shared range across all groups for this gene
                gene_min, gene_max = shared_range['min'], shared_range['max']
                gene_range = shared_range['range']
                if gene_range > 0:
                    padding = 0.05 * gene_range  # Smaller padding for shared scale
                    ax.set_xlim(gene_min - padding, gene_max + padding)
                else:
                    ax.set_xlim(gene_min - 0.5, gene_max + 0.5)
            else:
                # Fall back to individual data range
                data_range = data.max() - data.min()
                if data_range > 0:
                    padding = 0.1 * data_range
                    ax.set_xlim(data.min() - padding, data.max() + padding)
                else:
                    ax.set_xlim(data.min() - 0.5, data.max() + 0.5)
        else:
            # Vertical violin (expression on y-axis)
            parts = ax.violinplot([data], positions=[0.5], widths=[violin_width],
                                 vert=True, showmeans=False, showmedians=False, showextrema=False)

            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(alpha)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.5)

            ax.set_xlim(0, 1)
            # Set y-axis limits using shared range if provided, otherwise use data range
            if shared_range is not None:
                # Use shared range across all groups for this gene
                gene_min, gene_max = shared_range['min'], shared_range['max']
                gene_range = shared_range['range']
                if gene_range > 0:
                    padding = 0.05 * gene_range  # Smaller padding for shared scale
                    ax.set_ylim(gene_min - padding, gene_max + padding)
                else:
                    ax.set_ylim(gene_min - 0.5, gene_max + 0.5)
            else:
                # Fall back to individual data range
                data_range = data.max() - data.min()
                if data_range > 0:
                    padding = 0.1 * data_range
                    ax.set_ylim(data.min() - padding, data.max() + padding)
                else:
                    ax.set_ylim(data.min() - 0.5, data.max() + 0.5)

    except Exception as e:
        # If violin plot fails (e.g., not enough data for KDE), fall back to simple visualization
        warnings.warn(f"Violin plot failed for data with {len(data)} points, using fallback visualization")

        if horizontal:
            # Simple horizontal line plot as fallback
            ax.scatter(data, [0.5] * len(data), c=color, alpha=alpha, s=20)
            ax.set_ylim(0, 1)
            if len(data) > 0:
                if shared_range is not None:
                    gene_min, gene_max = shared_range['min'], shared_range['max']
                    gene_range = shared_range['range']
                    if gene_range > 0:
                        padding = 0.05 * gene_range
                        ax.set_xlim(gene_min - padding, gene_max + padding)
                    else:
                        ax.set_xlim(gene_min - 0.5, gene_max + 0.5)
                else:
                    data_range = data.max() - data.min()
                    if data_range > 0:
                        padding = 0.1 * data_range
                        ax.set_xlim(data.min() - padding, data.max() + padding)
                    else:
                        ax.set_xlim(data.min() - 0.5, data.max() + 0.5)
        else:
            # Simple vertical scatter as fallback
            ax.scatter([0.5] * len(data), data, c=color, alpha=alpha, s=20)
            ax.set_xlim(0, 1)
            if len(data) > 0:
                if shared_range is not None:
                    gene_min, gene_max = shared_range['min'], shared_range['max']
                    gene_range = shared_range['range']
                    if gene_range > 0:
                        padding = 0.05 * gene_range
                        ax.set_ylim(gene_min - padding, gene_max + padding)
                    else:
                        ax.set_ylim(gene_min - 0.5, gene_max + 0.5)
                else:
                    data_range = data.max() - data.min()
                    if data_range > 0:
                        padding = 0.1 * data_range
                        ax.set_ylim(data.min() - padding, data.max() + padding)
                    else:
                        ax.set_ylim(data.min() - 0.5, data.max() + 0.5)


def _add_gene_group_annotation(fig, axes, genes: List[str], marker_groups: Optional[Dict] = None,
                              marker_group_colors: Optional[Dict] = None,
                              annotation_height: float = 0.02,
                              invert_axes: bool = False) -> None:
    """
    Add colored annotation bar showing gene groupings.

    Parameters
    ----------
    fig : plt.Figure
        Figure object
    axes : np.ndarray
        Axes array
    genes : List[str]
        List of gene names
    marker_groups : Optional[Dict]
        Dictionary mapping group names to lists of genes
    marker_group_colors : Optional[Dict]
        Dictionary mapping group names to colors
    annotation_height : float
        Height of annotation bar relative to figure
    invert_axes : bool
        Whether axes are inverted
    """
    if marker_groups is None:
        return

    # Get group for each gene
    gene_to_group = {}
    for group_name, gene_list in marker_groups.items():
        for gene in gene_list:
            gene_to_group[gene] = group_name

    # Get colors for groups
    unique_groups = list(marker_groups.keys())
    if marker_group_colors is None:
        group_colors = dict(zip(unique_groups, sns.color_palette('Set1', len(unique_groups))))
    else:
        group_colors = marker_group_colors

    # Create annotation bar
    if invert_axes:
        # Groups on x-axis, genes on y-axis - annotation on left
        # For now, skip annotation in inverted mode (can implement later)
        return
    else:
        # Genes on x-axis, groups on y-axis - annotation at bottom
        n_genes = len(genes)
        n_cols = axes.shape[1]  # Number of subplot columns

        # Get position of bottom row of subplots - use the actual subplot positions
        bottom_axes = axes[-1, :]  # Bottom row

        # Calculate the exact positions to align with subplot columns
        subplot_positions = []
        for i in range(n_cols):
            pos = bottom_axes[i].get_position()
            subplot_positions.append((pos.x0, pos.x1))

        # Total span of subplots
        left_edge = subplot_positions[0][0]
        right_edge = subplot_positions[-1][1]
        total_width = right_edge - left_edge

        # Get the bottom position
        bottom_pos = bottom_axes[0].get_position().y0

        # Create annotation axis - positioned well below the entire plot
        # Use a fixed position at the bottom of the figure
        ann_y_pos = 0.02  # Fixed position at bottom of figure
        ann_ax = fig.add_axes([left_edge, ann_y_pos, total_width, annotation_height])

        # Draw colored rectangles for each gene, aligned with subplot columns
        # Calculate rectangle width and position to match subplot columns exactly
        n_genes = len(genes)
        col_width = total_width / n_genes  # Equal width for each gene column

        for i, gene in enumerate(genes):
            gene_group = gene_to_group.get(gene, 'Unknown')
            color = group_colors.get(gene_group, 'lightgray')

            # Position rectangle to align with gene column
            rect_x = i * col_width
            rect = Rectangle((rect_x, 0), col_width, 1,
                           facecolor=color, edgecolor='black', linewidth=0.5,
                           alpha=0.8)
            ann_ax.add_patch(rect)

        ann_ax.set_xlim(0, total_width)
        ann_ax.set_ylim(0, 1)
        ann_ax.set_xticks([])
        ann_ax.set_yticks([])

        # Remove all spines for clean look
        for spine in ann_ax.spines.values():
            spine.set_visible(False)


def stacked_vlnplot(adata, genes: List[str], group_by: str,
                   invert_axes: bool = False,
                   marker_groups: Optional[Dict] = None,
                   marker_group_colors: Optional[Dict] = None,
                   layer: Optional[str] = None,
                   group_order: Optional[List[str]] = None,
                   group_colors: Optional[Dict] = None,
                   show_gene_groups: bool = True,
                   figsize: Union[str, Tuple[float, float]] = 'auto',
                   violin_alpha: float = 0.7,
                   hide_axes: str = 'both',
                   title: Optional[str] = None,
                   title_fontsize: int = 14,
                   gene_label_fontsize: int = 10,
                   group_label_fontsize: int = 8,
                   gene_label_rotation: float = 0,
                   debug: bool = False) -> plt.Figure:
    """
    Create stacked violin plots arranged in a grid layout.

    Similar to dotplots but showing full expression distributions. Particularly useful
    for marker validation and cell annotation justification.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    genes : List[str]
        List of gene names to plot
    group_by : str
        Column name in adata.obs for grouping cells
    invert_axes : bool, default False
        If False (default): genes on x-axis, groups on y-axis (horizontal violins)
        If True: groups on x-axis, genes on y-axis (vertical violins)
    marker_groups : Optional[Dict], default None
        Dictionary mapping group names to lists of genes for color coding.
        E.g., {'T_cell': ['CD3D', 'CD8A'], 'B_cell': ['CD19', 'MS4A1']}
    marker_group_colors : Optional[Dict], default None
        Dictionary mapping group names to colors. If None, uses default palette.
    layer : Optional[str], default None
        Layer to use for expression values. If None, uses .X
    group_order : Optional[List[str]], default None
        Order of groups. If None, uses natural order from data.
    group_colors : Optional[Dict], default None
        Colors for cell groups. If None, uses default palette.
    show_gene_groups : bool, default True
        Whether to show colored annotation bar for gene groups
    figsize : Union[str, Tuple[float, float]], default 'auto'
        Figure size. If 'auto', calculates based on grid dimensions.
    violin_alpha : float, default 0.7
        Transparency of violin plots
    hide_axes : str, default 'both'
        Which axes to hide: 'x', 'y', 'both', or 'none'
    title : Optional[str], default None
        Main plot title
    title_fontsize : int, default 14
        Font size for main title
    gene_label_fontsize : int, default 10
        Font size for gene labels
    group_label_fontsize : int, default 8
        Font size for group labels
    gene_label_rotation : float, default 0
        Rotation angle for gene labels (degrees)

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Reshape data to long format
    df = _reshape_expression_data(adata, genes, group_by, layer)

    # Debug information
    if debug:
        print(f"Data shape: {df.shape}")
        print(f"Genes: {df['gene'].unique()}")
        print(f"Groups: {df[group_by].unique()}")

        # Check B cell expression specifically
        b_cell_data = df[df[group_by] == 'B_cells'] if 'B_cells' in df[group_by].unique() else None
        if b_cell_data is not None:
            print(f"\nB_cells subset shape: {b_cell_data.shape}")
            print("B_cells expression summary:")
            for gene in df['gene'].unique():
                gene_expr = b_cell_data[b_cell_data['gene'] == gene]['expression']
                if len(gene_expr) > 0:
                    print(f"  {gene}: mean={gene_expr.mean():.2f}, max={gene_expr.max():.2f}, n_cells={len(gene_expr)}")

    # Get unique genes and groups
    unique_genes = df['gene'].cat.categories.tolist()
    unique_groups = df[group_by].unique().tolist()

    # Apply group order if provided
    if group_order is not None:
        unique_groups = [g for g in group_order if g in unique_groups]

    # Get colors for groups
    group_color_list = _get_colors_from_dict(group_colors, unique_groups)
    group_color_dict = dict(zip(unique_groups, group_color_list))

    # Create grid
    fig, axes = _create_stacked_grid(len(unique_genes), len(unique_groups),
                                    figsize, invert_axes)

    # Calculate shared expression ranges for each gene to ensure proper scaling
    gene_expression_ranges = {}
    for gene in unique_genes:
        gene_data = df[df['gene'] == gene]['expression']
        gene_expression_ranges[gene] = {
            'min': gene_data.min(),
            'max': gene_data.max(),
            'range': gene_data.max() - gene_data.min()
        }

    # Plot violins
    for i, group in enumerate(unique_groups):
        for j, gene in enumerate(unique_genes):
            if invert_axes:
                ax = axes[j, i]  # genes on rows, groups on cols
            else:
                ax = axes[i, j]  # groups on rows, genes on cols

            # Get data for this gene-group combination
            subset = df[(df['gene'] == gene) & (df[group_by] == group)]
            data = subset['expression'].values

            # Get shared expression range for this gene
            gene_range = gene_expression_ranges[gene]

            # Plot violin with shared expression range
            _plot_grid_violin(ax, data, horizontal=not invert_axes,
                            color=group_color_dict[group], alpha=violin_alpha,
                            shared_range=gene_range)

            # Hide axes as requested
            if hide_axes in ['x', 'both']:
                ax.set_xticks([])
                ax.set_xlabel('')
            if hide_axes in ['y', 'both']:
                ax.set_yticks([])
                ax.set_ylabel('')

            # Remove spines for clean look
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Add labels
    if not invert_axes:
        # Genes on x-axis (columns), groups on y-axis (rows)
        # Add gene labels to top
        for j, gene in enumerate(unique_genes):
            axes[0, j].set_title(gene, fontsize=gene_label_fontsize, pad=10,
                                rotation=gene_label_rotation)

        # Add group labels to left
        for i, group in enumerate(unique_groups):
            axes[i, 0].set_ylabel(group, fontsize=group_label_fontsize, rotation=0,
                                 ha='right', va='center')
    else:
        # Groups on x-axis (columns), genes on y-axis (rows)
        # Add group labels to top
        for i, group in enumerate(unique_groups):
            axes[0, i].set_title(group, fontsize=group_label_fontsize, pad=10)

        # Add gene labels to left
        for j, gene in enumerate(unique_genes):
            axes[j, 0].set_ylabel(gene, fontsize=gene_label_fontsize,
                                 rotation=gene_label_rotation,
                                 ha='right', va='center')

    # Add gene group annotation if requested
    if show_gene_groups and marker_groups is not None:
        _add_gene_group_annotation(fig, axes, unique_genes, marker_groups,
                                  marker_group_colors, invert_axes=invert_axes)

    # Add main title
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize, y=0.98)

    return fig