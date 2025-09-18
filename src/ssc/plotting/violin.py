"""
Violin plot functions for single-cell RNA sequencing data visualization.
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import warnings
import time
from collections import OrderedDict
import hashlib


# Global DE cache with LRU eviction
_de_cache = OrderedDict()
_de_cache_limit = 50  # Default limit for cached DE results


def set_de_cache_limit(limit):
    """Set the maximum number of DE results to cache."""
    global _de_cache_limit
    _de_cache_limit = limit
    _evict_cache()


def clear_de_cache():
    """Clear all cached DE results."""
    global _de_cache
    _de_cache.clear()
    print("🗑️ DE cache cleared")


def _evict_cache():
    """Remove oldest cache entries if over limit."""
    global _de_cache
    while len(_de_cache) > _de_cache_limit:
        oldest_key = next(iter(_de_cache))
        del _de_cache[oldest_key]


def _generate_cache_key(adata, groupby, group1, group2, idx1, idx2, mode, delta, batch_correction):
    """Generate unique cache key for DE comparison."""
    # Create data fingerprint to detect if adata changed
    if groupby is not None and groupby in adata.obs.columns:
        # Convert to string to handle categorical data safely
        try:
            data_values = adata.obs[groupby].astype(str).to_numpy()
            data_hash = hashlib.md5(data_values.tobytes()).hexdigest()[:8]
        except:
            # Fallback if there are issues with the data
            data_hash = hashlib.md5(str(adata.shape).encode()).hexdigest()[:8]
    else:
        data_hash = hashlib.md5(str(adata.shape).encode()).hexdigest()[:8]

    if idx1 is not None:
        # Custom cell indices
        idx1_hash = hashlib.md5(str(sorted(idx1)).encode()).hexdigest()[:8]
        idx2_hash = hashlib.md5(str(sorted(idx2)).encode()).hexdigest()[:8]
        key = f"custom_{idx1_hash}_vs_{idx2_hash}"
    else:
        # Group-based comparison
        key = f"{groupby}_{group1}_vs_{group2}"

    return f"{key}_{mode}_{delta}_{batch_correction}_{data_hash}"


def _get_or_compute_de_working(scvi_model, adata, groupby=None, group1=None, group2=None,
                               subset_data=None, mode='change', delta=0.25, **kwargs):
    """Get DE results using the working approach from script 07b."""

    # Generate cache key for this specific comparison
    if subset_data is not None:
        # Use subset data approach (working method)
        data_hash = hashlib.md5(str(subset_data.shape).encode()).hexdigest()[:8]
        cache_key = f"subset_{groupby}_{group1}_vs_{group2}_{mode}_{delta}_{data_hash}"

        if cache_key in _de_cache:
            de_results = _de_cache.pop(cache_key)
            _de_cache[cache_key] = de_results
            print(f"✅ Using cached DE results for {group1} vs {group2}")
            return de_results

        # Compute using working approach: subset data + change mode
        print(f"🔄 Computing DE for {group1} vs {group2} (working approach)...")
        start_time = time.time()

        try:
            # Use subset data directly with groupby (1-vs-all approach from script 07b)
            de_results = scvi_model.differential_expression(
                subset_data,          # Use subset directly (working approach)
                groupby=groupby,      # This creates "group1 vs Rest", "group2 vs Rest", etc.
                mode=mode            # Use 'change' mode (working approach)
            )

            # Cache the results
            _de_cache[cache_key] = de_results
            _evict_cache()

            elapsed = time.time() - start_time
            print(f"✅ DE computation completed in {elapsed:.1f} seconds")
            return de_results

        except Exception as e:
            print(f"❌ DE computation failed: {str(e)}")
            raise

    else:
        # Fallback to old method if no subset provided
        return _get_or_compute_de_old(scvi_model, adata, groupby, group1, group2, mode, delta, **kwargs)


def _get_or_compute_de_old(scvi_model, adata, groupby=None, group1=None, group2=None,
                          mode='vanilla', delta=0.25, batch_correction=False, **kwargs):
    """Original DE method - kept as fallback."""
    cache_key = _generate_cache_key(adata, groupby, group1, group2, None, None,
                                   mode, delta, batch_correction)

    if cache_key in _de_cache:
        de_results = _de_cache.pop(cache_key)
        _de_cache[cache_key] = de_results
        print(f"✅ Using cached DE results for {group1} vs {group2}")
        return de_results

    print(f"🔄 Computing DE for {group1} vs {group2} (old approach)...")
    start_time = time.time()

    try:
        de_results = scvi_model.differential_expression(
            adata=adata,
            groupby=groupby,
            group1=group1,
            group2=group2,
            mode=mode,
            delta=delta,
            batch_correction=batch_correction,
            **kwargs
        )

        _de_cache[cache_key] = de_results
        _evict_cache()

        elapsed = time.time() - start_time
        print(f"✅ DE computation completed in {elapsed:.1f} seconds")
        return de_results

    except Exception as e:
        print(f"❌ DE computation failed: {str(e)}")
        raise


def _extract_pairwise_comparison(de_results_1vs_all, gene, group1, group2):
    """Extract pairwise comparison from 1-vs-all DE results."""
    try:
        # Get group1 vs Rest and group2 vs Rest results
        group1_vs_rest = de_results_1vs_all[de_results_1vs_all.comparison == f"{group1} vs Rest"]
        group2_vs_rest = de_results_1vs_all[de_results_1vs_all.comparison == f"{group2} vs Rest"]

        if gene not in group1_vs_rest.index or gene not in group2_vs_rest.index:
            return None

        # Get gene-specific results
        gene1_stats = group1_vs_rest.loc[gene]
        gene2_stats = group2_vs_rest.loc[gene]

        # Calculate correct pairwise statistics
        # Get the actual expression levels for direct comparison
        scale1 = gene1_stats.get('scale1', 0)  # group1 mean expression
        scale2 = gene2_stats.get('scale1', 0)  # group2 mean expression

        # Calculate proper log fold change: log2(group2/group1)
        # Positive LFC means group2 > group1, negative means group2 < group1
        if scale1 > 0 and scale2 > 0:
            lfc_pairwise = np.log2(scale2 / scale1)
        else:
            lfc_pairwise = 0

        # For probability, use the maximum evidence that either group differs from baseline
        proba_de = max(gene1_stats['proba_de'], gene2_stats['proba_de'])

        # Create proper pairwise result
        pairwise_stats = pd.Series({
            'proba_de': proba_de,
            'lfc_mean': lfc_pairwise,  # Corrected: direct group2/group1 ratio
            'bayes_factor': max(gene1_stats['bayes_factor'], gene2_stats['bayes_factor']),
            'comparison': f"{group1} vs {group2}",
            'scale1': scale1,  # group1 expression level
            'scale2': scale2   # group2 expression level
        })

        return pairwise_stats

    except Exception as e:
        print(f"⚠️ Failed to extract pairwise comparison for {gene}: {str(e)}")
        return None


def _extract_de_statistics(gene_stats, mode='change'):
    """Extract statistics from scVI DE results, handling different modes."""
    if gene_stats is None or gene_stats.empty:
        return None, None

    # Now defaults to change mode (working approach)
    if mode == 'change' or 'proba_de' in gene_stats:
        # Change mode has direct proba_de and lfc_mean (working approach)
        proba_de = gene_stats.get('proba_de', 0)
        lfc_mean = gene_stats.get('lfc_mean', 0)
        return proba_de, lfc_mean

    elif mode == 'vanilla':
        # Legacy vanilla mode support
        bayes_factor = gene_stats.get('bayes_factor', 0)

        # Convert log Bayes factor to significance (corrected thresholds)
        if bayes_factor >= 2.3:      # log(10)
            significance = 0.95
        elif bayes_factor >= 1.1:    # log(3)
            significance = 0.8
        elif bayes_factor >= 0:      # log(1)
            significance = 0.6
        else:
            significance = 0.0

        # Calculate log fold change from scale ratios
        scale1 = gene_stats.get('scale1', 1)
        scale2 = gene_stats.get('scale2', 1)
        if scale1 > 0 and scale2 > 0:
            lfc = np.log2(scale2 / scale1)
        else:
            lfc = 0

        return significance, lfc

    else:
        return 0, 0


def _get_star_annotation(significance, thresholds=(0.6, 0.8, 0.95)):
    """Convert significance value to star annotation."""
    if significance >= thresholds[2]:
        return "***"
    elif significance >= thresholds[1]:
        return "**"
    elif significance >= thresholds[0]:
        return "*"
    else:
        return ""


def _add_statistical_annotation(ax, x1, x2, y_position, gene_stats, detailed_stats=False,
                               proba_de_thresholds=(0.6, 0.8, 0.95), de_mode='vanilla'):
    """Add statistical annotation between two x positions."""
    if gene_stats is None or gene_stats.empty:
        return

    # Extract statistics based on DE mode
    significance, lfc = _extract_de_statistics(gene_stats, mode=de_mode)

    if significance is None:
        return

    # Get star annotation
    stars = _get_star_annotation(significance, proba_de_thresholds)

    if not stars:
        return  # No significant difference to annotate

    # Draw horizontal line
    line_height = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x2], [y_position, y_position], 'k-', linewidth=1)
    ax.plot([x1, x1], [y_position - line_height/2, y_position + line_height/2], 'k-', linewidth=1)
    ax.plot([x2, x2], [y_position - line_height/2, y_position + line_height/2], 'k-', linewidth=1)

    # Add annotation text
    x_center = (x1 + x2) / 2
    if detailed_stats:
        if de_mode == 'vanilla':
            bayes_factor = gene_stats.get('bayes_factor', 0)
            annotation = f"{stars}\nBF={bayes_factor:.2f}"
        else:
            annotation = f"{stars}\nP={significance:.2f}"
        if abs(lfc) >= 0.1:  # Only show LFC if meaningful
            annotation += f"\nLFC={lfc:.1f}"
    else:
        annotation = stars

    ax.text(x_center, y_position + line_height, annotation, ha='center', va='bottom',
            fontsize=14, fontweight='bold')


def _process_statistical_effects(adata, gene, scvi_model, group_effects, split_effects,
                                group_by, split_by, de_mode, de_delta, groups, splits):
    """Process and compute statistical effects using the working approach from script 07b."""
    if not scvi_model:
        return {}

    gene_statistics = {}

    # Process group effects using working approach (subset data + change mode)
    if group_effects:
        print(f"🔄 Processing group effects using working approach...")
        try:
            # Use the entire dataset for group comparisons (working approach)
            de_results_1vs_all = _get_or_compute_de_working(
                scvi_model, adata,
                groupby=group_by,
                subset_data=adata,  # Use full data for group comparisons
                mode='change'       # Use change mode (working approach)
            )

            # Extract pairwise comparisons from 1-vs-all results
            for comparison in group_effects:
                if isinstance(comparison, tuple) and len(comparison) == 2:
                    group1, group2 = comparison
                    try:
                        pairwise_stats = _extract_pairwise_comparison(
                            de_results_1vs_all, gene, group1, group2
                        )
                        if pairwise_stats is not None:
                            gene_statistics[f"group_{group1}_vs_{group2}"] = pairwise_stats
                            print(f"   ✅ {group1} vs {group2}: proba_de={pairwise_stats['proba_de']:.3f}")
                        else:
                            print(f"   ⚠️ {group1} vs {group2}: No data for {gene}")
                    except Exception as e:
                        print(f"   ❌ {group1} vs {group2}: {str(e)}")

        except Exception as e:
            print(f"❌ Failed to compute group effects: {str(e)}")

    # Process split effects using working approach
    if split_effects and split_by:
        print(f"🔄 Processing split effects using working approach...")
        for comparison in split_effects:
            if isinstance(comparison, tuple) and len(comparison) == 2:
                split1, split2 = comparison
                try:
                    # Create subset containing only the split conditions (working approach)
                    split_mask = adata.obs[split_by].isin([split1, split2])
                    adata_split_subset = adata[split_mask].copy()

                    if adata_split_subset.n_obs < 10:
                        print(f"   ⚠️ {split1} vs {split2}: Too few cells ({adata_split_subset.n_obs})")
                        continue

                    # Use working approach: subset data + change mode + groupby
                    de_results_split = _get_or_compute_de_working(
                        scvi_model, adata,
                        groupby=split_by,
                        group1=split1,
                        group2=split2,
                        subset_data=adata_split_subset,
                        mode='change'
                    )

                    # Extract pairwise comparison
                    pairwise_stats = _extract_pairwise_comparison(
                        de_results_split, gene, split1, split2
                    )
                    if pairwise_stats is not None:
                        gene_statistics[f"split_{split1}_vs_{split2}"] = pairwise_stats
                        print(f"   ✅ {split1} vs {split2}: proba_de={pairwise_stats['proba_de']:.3f}")
                    else:
                        print(f"   ⚠️ {split1} vs {split2}: No data for {gene}")

                except Exception as e:
                    print(f"   ❌ {split1} vs {split2}: {str(e)}")

    return gene_statistics


def _draw_empty_placeholder_clean(ax, x_pos, text="No data"):
    """Draw a clean text-only placeholder for missing data"""
    ax.text(x_pos, ax.get_ylim()[1] * 0.1, text, ha='center', va='center',
            fontsize=8, color='gray', style='italic')


def _draw_empty_split_placeholder(ax, x_pos, width, color):
    """Draw a small empty placeholder for missing split data"""
    y_center = ax.get_ylim()[1] * 0.05
    violin_height = ax.get_ylim()[1] * 0.02

    ellipse = Ellipse((x_pos, y_center), width=width*0.5, height=violin_height,
                     fill=False, edgecolor=color, linestyle=':', alpha=0.3)
    ax.add_patch(ellipse)


def _create_single_plot_layout(figsize):
    """Create single plot layout and return consistent format"""
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    return fig, [(ax1, ax2, None, None)]  # (ax1, ax2, row, col)


def _create_faceted_plot_layout(n_rows, n_cols, figsize):
    """Create faceted plot layout and return consistent format"""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    ax_pairs = []
    for row in range(n_rows):
        for col in range(n_cols):
            ax1 = axes[row, col]
            ax2 = ax1.twinx()
            ax_pairs.append((ax1, ax2, row, col))
    return fig, ax_pairs


def _get_colors_from_dict(color_dict, categories, default_palette, color_type):
    """Get colors with fallback to default palette if mismatch"""
    if color_dict is None:
        return default_palette

    # Check if all categories are covered
    missing_categories = set(categories) - set(color_dict.keys())
    extra_categories = set(color_dict.keys()) - set(categories)

    if missing_categories or extra_categories:
        if missing_categories:
            print(f"⚠️  {color_type}: Missing colors for {missing_categories}")
        if extra_categories:
            print(f"⚠️  {color_type}: Unused colors for {extra_categories}")
        print(f"   Using default palette for all {color_type.lower()}")
        return default_palette

    # All good - return colors in category order
    return [color_dict[cat] for cat in categories]

def _detect_scvi_data(adata, expression_data, layer_name=None):
    """Detect if data appears to be scVI-normalized"""
    zero_fraction = (expression_data == 0).mean()
    has_scvi_metadata = any('scvi' in key.lower() for key in adata.uns.keys())
    small_positive_fraction = ((expression_data > 0) & (expression_data < 0.1)).mean()

    likely_scvi = (
        zero_fraction < 0.05 and  # Very few zeros
        small_positive_fraction > 0.2 and  # Many small positive values
        (has_scvi_metadata or layer_name is None)
    )

    return likely_scvi, zero_fraction

def _suggest_percent_expressing_layer(adata):
    """Suggest best layer for % expressing calculation"""
    candidate_layers = ['raw', 'uncorrected', 'counts', 'X_original']

    for layer in candidate_layers:
        if layer in adata.layers:
            layer_data = adata.layers[layer]
            if hasattr(layer_data, 'toarray'):
                sample_data = layer_data[:1000].toarray().flatten()  # Sample for speed
            else:
                sample_data = layer_data[:1000].flatten()

            zero_frac = (sample_data == 0).mean()
            if zero_frac > 0.1:  # Reasonable sparsity
                return layer

    return None


def vlnplot(adata, gene, group_by, 
                    title=None, 
                    layer=None,
                    percent_expressing_layer=None,
                    split_by=None,
                    facet_by=None,
                    facet_col=None,
                    group_order=None,
                    split_order=None,
                    facet_order=None,
                    facet_col_order=None,
                    group_colors=None,
                    split_colors=None,
                    jitter_points=True,
                    jitter_dot_size=12,
                    plot_mean=True,
                    show_fraction=True,
                    show_stats=True,
                    show_legend=True,
                    title_fontsize=14,           # Main plot title
                    subtitle_fontsize=9,         # Individual subplot titles  
                    ylabel_fontsize=10,          # Y-axis labels (left side)
                    ylabel_mean_fontsize=10,     # Mean expression y-axis labels (right side)
                    xlabel_fontsize=8,           # X-axis group labels (existing, but now more explicit)
                    axis_tick_fontsize=8,        # Axis tick numbers
                    legend_fontsize=8,           # Split legend
                    number_fontsize=6,           # Cell count/fraction numbers (existing)
                    number_decimal_places=2,
                    xlabel_rotation=45,
                    xlabel_ha='right',
                    figsize=(12, 8),
                    facet_figsize=None,
                    facet_ncols=None,
                    mean_color='black',
                    free_y=True,
                    free_mean_y=False,
                    ylim=None):
    """
    Create violin plots for single-cell RNA sequencing data with comprehensive customization options.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing single-cell expression data
    gene : str
        Gene name to plot (must exist in adata.var_names)
    group_by : str
        Column name in adata.obs for grouping cells on x-axis
    title : str, optional
        Main plot title. If None, uses gene name
    layer : str, optional
        Layer to use for expression data. If None, uses adata.X
    split_by : str, optional
        Column name in adata.obs for split violin plots (e.g., 'treatment', 'genotype', 'stimulus')
    facet_by : str, optional
        Column name in adata.obs for faceting by columns
    facet_col : str, optional
        Column name in adata.obs for faceting by rows
    group_order : list, optional
        Custom order for group_by categories
    split_order : list, optional
        Custom order for split_by categories
    facet_order : list, optional
        Custom order for facet_by categories
    facet_col_order : list, optional
        Custom order for facet_col categories
    group_colors : dict, optional
        Dictionary mapping group names to colors (ignored when split_by is used)
    split_colors : dict, optional
        Dictionary mapping split names to colors (overrides group_colors when splitting)
    jitter_points : bool, default True
        Whether to show individual data points
    jitter_dot_size : int, default 12
        Size of jitter points
    plot_mean : bool, default True
        Whether to show mean expression dots on secondary y-axis
    show_fraction : bool, default True
        Whether to show cell count and fraction expressing below x-axis
    show_stats : bool, default True
        Whether to show cell statistics
    show_legend : bool, default True
        Whether to show legend for split violins
    title_fontsize : int, default 14
        Font size for main plot title
    subtitle_fontsize : int, default 9
        Font size for individual subplot titles
    ylabel_fontsize : int, default 10
        Font size for y-axis labels (left side)
    ylabel_mean_fontsize : int, default 10
        Font size for mean expression y-axis labels (right side)
    xlabel_fontsize : int, default 8
        Font size for x-axis group labels
    axis_tick_fontsize : int, default 8
        Font size for axis tick numbers
    legend_fontsize : int, default 8
        Font size for split legend
    number_fontsize : int, default 6
        Font size for cell count/fraction numbers
    number_decimal_places : int, default 2
        Number of decimal places for fraction expressing
    xlabel_rotation : float, default 45
        Rotation angle for x-axis group labels in degrees
    xlabel_ha : str, default 'right'
        Horizontal alignment for x-axis group labels ('left', 'center', 'right')
    figsize : tuple, default (12, 8)
        Figure size for single plots
    facet_figsize : tuple, optional
        Figure size for faceted plots. If None, calculated automatically
    facet_ncols : int, optional
    Number of columns for facet_by grid layout. If None, uses horizontal layout (default).
    When specified, arranges facet_by categories in an n_cols × n_rows grid with padding
    for incomplete grids. If facet_col is also used, each facet_col category gets its
    own n_cols × n_rows block stacked vertically.
    Example: 4 subjects with facet_ncols=2 creates 2×2 grid; with facet_col='tissue'
    (2 tissues), creates 4×2 total grid (two 2×2 blocks stacked).
    mean_color : str, default 'black'
        Color for mean expression dots and axis
    free_y : bool, default True
        If True, allow independent gene expression y-axis scaling for each subplot
    free_mean_y : bool, default False
        If True, allow independent mean expression y-axis scaling for each subplot
    ylim : tuple, optional
        Y-axis limits for gene expression (left axis) as (min, max). 
        Overrides free_y behavior when specified. Mean expression axis unaffected.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    
    Examples
    --------
    >>> # Basic violin plot
    >>> fig = ssc.vlnplot(adata, 'GNLY', 'cell_type')
    
    >>> # Split violin with custom colors
    >>> fig = ssc.vlnplot(adata, 'GNLY', 'condition', split_by='treatment',
    ...                   split_colors={'pre': 'lightblue', 'dupi': 'darkblue'})
    
    >>> # Faceted plot with custom ordering
    >>> fig = ssc.vlnplot(adata, 'GNLY', 'condition', 
    ...                   facet_by='subject', facet_col='cell_type',
    ...                   group_order=['Nonlesional', 'SADBE', 'Metal'])
    
    >>> # Focus on expression range with ylim
    >>> fig = ssc.vlnplot(adata, 'IL13', 'cell_type',
    ...                   ylim=(0, 10))  # Crop outliers for better violin detail
    """

    # Extract and validate data
    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene} not found in dataset")

    # Get expression data for plotting
    if layer is not None:
        if layer not in adata.layers.keys():
            raise ValueError(f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}")
        expression_data = adata[:, gene].layers[layer].toarray().flatten()
        data_source = f"layer '{layer}'"
    else:
        expression_data = adata[:, gene].X.toarray().flatten()
        data_source = "adata.X (default)"

    # Detect scVI data and handle % expressing calculation
    is_scvi, zero_frac = _detect_scvi_data(adata, expression_data, layer)

    # Handle percent expressing layer logic
    if percent_expressing_layer is None and is_scvi:
        suggested_layer = _suggest_percent_expressing_layer(adata)
        if suggested_layer:
            print(f"🔍 Detected scVI-like data ({zero_frac:.1%} zeros in expression data)")
            print(f"💡 Using '{suggested_layer}' layer for % expressing calculation")
            print(f"   This shows scVI means with original sparsity patterns")
            percent_expressing_layer = suggested_layer
        else:
            print(f"⚠️  Detected scVI-like data but no raw layer found for % expressing")
            percent_expressing_layer = None

    # Get data for percent expressing calculation
    if percent_expressing_layer is not None:
        if percent_expressing_layer not in adata.layers.keys():
            raise ValueError(f"Percent expressing layer '{percent_expressing_layer}' not found")
        # Store as DataFrame for easier indexing
        percent_df = pd.DataFrame({
            'percent_expr': adata[:, gene].layers[percent_expressing_layer].toarray().flatten()
        }, index=adata.obs.index)
        percent_source = f"layer '{percent_expressing_layer}'"
    else:
        # Store as DataFrame for easier indexing
        percent_df = pd.DataFrame({
            'percent_expr': expression_data
        }, index=adata.obs.index)
        percent_source = data_source

    # Build plot data
    plot_data = pd.DataFrame({
        'group': adata.obs[group_by].values,
        'expression': expression_data
    }, index=adata.obs.index)

    # Also subset the percent_df to match the current adata subset
    percent_df = percent_df.loc[adata.obs.index]

    if split_by is not None:
        plot_data['split'] = adata.obs[split_by].values

    if facet_by is not None:
        plot_data['facet_by'] = adata.obs[facet_by].values

    if facet_col is not None:
        plot_data['facet_col'] = adata.obs[facet_col].values

    print(f"📊 Using data from: {data_source}")
    print(f"📈 Expression range: {expression_data.min():.3f} - {expression_data.max():.3f}")
    print(f"🎨 Mean color: {mean_color}")
    print(f"🔓 Free gene expression axis: {free_y}")
    print(f"🔓 Free mean expression axis: {free_mean_y}")
    print(f"📋 Fraction labels: {show_fraction} (font size: {number_fontsize})")
    print(f"🏷️  X-axis labels: size {xlabel_fontsize}, rotation {xlabel_rotation}°, align {xlabel_ha}")
    if percent_expressing_layer is not None:
        print(f"📊 % expressing from: {percent_source}")
        print(f"🔍 Dual-source mode: scVI means + original sparsity")

    if expression_data.min() < 0:
        print(f"⚠️  Warning: Found negative expression values (min: {expression_data.min():.3f})")

    # Determine category orders with validation
    def _get_category_order(data_col, custom_order, param_name):
        """Get category order with validation"""
        available_categories = set(data_col.unique())

        if custom_order is not None:
            custom_set = set(custom_order)
            missing_in_custom = available_categories - custom_set
            missing_in_data = custom_set - available_categories

            if missing_in_data:
                print(f"⚠️  {param_name}: Categories {missing_in_data} not found in data")
            if missing_in_custom:
                print(f"⚠️  {param_name}: Categories {missing_in_custom} in data but not in custom order")

            # Use only categories that exist in both data and custom order
            valid_order = [cat for cat in custom_order if cat in available_categories]
            print(f"📋 {param_name} order: {valid_order}")
            return valid_order
        else:
            # Default alphabetical order
            default_order = sorted(available_categories)
            print(f"📋 {param_name} order (alphabetical): {default_order}")
            return default_order

    # Apply category ordering
    global_groups = _get_category_order(plot_data['group'], group_order, 'Groups')

    if split_by is not None:
        global_splits = _get_category_order(plot_data['split'], split_order, 'Splits')
    else:
        global_splits = None

    # Determine layout with ordered categories
    if facet_by is not None or facet_col is not None:
        # Faceted plot
        if facet_by is not None:
            facet_categories = _get_category_order(plot_data['facet_by'], facet_order, 'Facet')
            n_facets = len(facet_categories)

            # Calculate facet_by grid dimensions
            if facet_ncols is not None:
                facet_cols = min(facet_ncols, n_facets)
                facet_rows = math.ceil(n_facets / facet_cols)
            else:
                facet_cols = n_facets  # Default: horizontal layout
                facet_rows = 1
        else:
            facet_categories = [None]
            facet_cols = 1
            facet_rows = 1
            n_facets = 1

        if facet_col is not None:
            facet_col_categories = _get_category_order(plot_data['facet_col'], facet_col_order, 'Facet columns')
            n_rows = facet_rows * len(facet_col_categories)  # Stack grids vertically
            n_cols = facet_cols
        else:
            facet_col_categories = [None]
            n_rows = facet_rows
            n_cols = facet_cols

        # Better default figure size
        if facet_figsize is not None:
            figsize = facet_figsize
        else:
            figsize = (4 * n_cols, 4 * n_rows)  # Adjust for new grid

        fig, ax_pairs = _create_faceted_plot_layout(n_rows, n_cols, figsize)
        is_faceted = True
    else:
        # Single plot
        fig, ax_pairs = _create_single_plot_layout(figsize)
        facet_categories = [None]
        facet_col_categories = [None]
        n_rows = n_cols = 1
        is_faceted = False

    # Always calculate global y-axis limits
    y_max_global = plot_data['expression'].max()
    y_min_global = max(0, plot_data['expression'].min())
    y_range_global = y_max_global - y_min_global
    global_y_limit_min = max(0, y_min_global - (y_range_global * 0.05))
    global_y_limit_max = y_max_global + (y_range_global * 0.1)

    if not free_y:
        print(f"🎯 Global gene expression y-axis: {global_y_limit_min:.3f} to {global_y_limit_max:.3f}")
    else:
        print(f"🎯 Free y-axis mode: each subplot will calculate independent limits")

    # Storage for scaling
    all_means = []
    subplot_y_data = {}
    subplot_mean_data = {}

    # Initialize critical variables to prevent reference errors
    global_max_expression = 0
    mean_axis_scale_factor = 1.0

    # Plot each subplot
    for idx, (ax1, ax2, row, col) in enumerate(ax_pairs):

        # Determine which data subset to use with new grid logic
        if is_faceted:
            if facet_col is not None and facet_by is not None:
                # Calculate which tissue and subject from grid position
                tissue_idx = row // facet_rows
                subject_row = row % facet_rows
                subject_idx = subject_row * facet_cols + col

                # Skip empty positions
                if subject_idx >= n_facets:
                    ax1.axis('off')
                    ax2.axis('off')
                    continue

                facet_col_cat = facet_col_categories[tissue_idx]
                facet_cat = facet_categories[subject_idx]
            elif facet_by is not None:
                # Single faceting with grid
                subject_idx = row * facet_cols + col
                if subject_idx >= n_facets:
                    ax1.axis('off')
                    ax2.axis('off')
                    continue
                facet_cat = facet_categories[subject_idx]
                facet_col_cat = None
            else:
                facet_col_cat = facet_col_categories[row] if facet_col_categories[0] is not None else None
                facet_cat = None

            subset_data = plot_data.copy()
            if facet_col_cat is not None:
                subset_data = subset_data[subset_data['facet_col'] == facet_col_cat]
            if facet_cat is not None:
                subset_data = subset_data[subset_data['facet_by'] == facet_cat]
        else:
            subset_data = plot_data
            facet_col_cat = facet_cat = None

        groups = global_groups
        subplot_means = []
        subplot_expressions = []

        if split_by is None:
            # Regular violins - use group colors
            colors = _get_colors_from_dict(group_colors, global_groups,
                                        sns.color_palette("Set2", len(global_groups)),
                                        "Group colors")

            for i, group in enumerate(groups):
                group_expr = subset_data[subset_data['group'] == group]['expression'].values

                if len(group_expr) > 0:
                    subplot_expressions.extend(group_expr)

                    # Create violin
                    parts = ax1.violinplot([group_expr], positions=[i],
                                        showmeans=False, showmedians=False, showextrema=False)

                    for pc in parts['bodies']:
                        pc.set_facecolor(colors[i % len(colors)])
                        pc.set_alpha(0.6)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(1)

                    if jitter_points:
                        x_jitter = np.random.normal(i, 0.1, size=len(group_expr))
                        violin_color = colors[i % len(colors)]
                        if isinstance(violin_color, str):
                            violin_color = mcolors.to_rgb(violin_color)
                        darker_color = tuple(c * 0.7 for c in violin_color[:3])  # 70% darker
                        ax1.scatter(x_jitter, group_expr,
                                c=[darker_color], s=jitter_dot_size, alpha=0.8, edgecolors='none')


                      # Calculate mean of expressing cells only
                    if percent_expressing_layer is not None:
                        # Use raw layer to identify expressing cells
                        percent_group_indices = subset_data[subset_data['group'] == group].index
                        # Use DataFrame indexing to get group data
                        group_indices = subset_data[subset_data['group'] == group].index
                        percent_group_data = percent_df.loc[group_indices, 'percent_expr']
                        expressing_mask = percent_group_data > 0

                        if expressing_mask.sum() > 0:
                            # Mean of only expressing cells (using scVI-normalized values)
                            group_mean = group_expr[expressing_mask].mean()
                        else:
                            group_mean = 0  # No expressing cells
                    else:
                        # Fallback: traditional mean if no raw layer
                        expressing_mask = group_expr > 0
                        if expressing_mask.sum() > 0:
                            group_mean = group_expr[expressing_mask].mean()
                        else:
                            group_mean = 0

                    subplot_means.append(group_mean)

                    if plot_mean:
                        ax2.scatter(i, group_mean, c=mean_color, s=60, marker='o',
                                edgecolors='white', linewidth=1, zorder=10)
                else:
                    _draw_empty_placeholder_clean(ax1, i, "No data")
                    subplot_means.append(0)

            # Cell statistics positioning - closer to x-axis
            if show_fraction:
                for i, group in enumerate(groups):
                    group_data = subset_data[subset_data['group'] == group]['expression']
                    if len(group_data) > 0:
                        n_cells = len(group_data)
                        if percent_expressing_layer is not None:
                            # Use percent_df for % expressing calculation
                            group_indices = subset_data[subset_data['group'] == group].index
                            percent_group_data = percent_df.loc[group_indices, 'percent_expr']
                            frac_expressing = (percent_group_data > 0).mean()
                        else:
                            frac_expressing = (group_data > 0).mean()

                        # Cell count - positioned just below x-axis
                        ax1.text(i, -0.08, f'{n_cells}', ha='center', va='top',
                                fontsize=number_fontsize, weight='bold',
                                transform=ax1.get_xaxis_transform())
                        # Fraction expressing - below cell count
                        ax1.text(i, -0.16, f'{frac_expressing:.{number_decimal_places}f}',
                                ha='center', va='top', fontsize=number_fontsize-1,
                                transform=ax1.get_xaxis_transform())
                    else:
                        # Empty data indicators
                        ax1.text(i, -0.08, '0', ha='center', va='top',
                                fontsize=number_fontsize, weight='bold', color='gray',
                                transform=ax1.get_xaxis_transform())
                        ax1.text(i, -0.16, '--', ha='center', va='top',
                                fontsize=number_fontsize-1, color='gray',
                                transform=ax1.get_xaxis_transform())

        else:
            # Split violin plot with custom colors
            n_splits = len(global_splits)
            violin_width = 0.8
            split_width = violin_width / n_splits

            # Use split colors when splitting
            split_colors_palette = _get_colors_from_dict(split_colors, global_splits,
                                                        sns.color_palette("Set3", len(global_splits)),
                                                        "Split colors")

            mean_positions = []
            mean_values = []

            for i, group in enumerate(groups):
                group_data = subset_data[subset_data['group'] == group]
                group_has_data = False

                for j, split_cat in enumerate(global_splits):
                    split_data = group_data[group_data['split'] == split_cat]['expression'].values
                    x_pos = i - (violin_width/2) + (j + 0.5) * split_width

                    if len(split_data) > 0:
                        group_has_data = True
                        subplot_expressions.extend(split_data)

                        parts = ax1.violinplot([split_data], positions=[x_pos],
                                            widths=[split_width * 0.8],
                                            showmeans=False, showmedians=False, showextrema=False)

                        for pc in parts['bodies']:
                            pc.set_facecolor(split_colors_palette[j])
                            pc.set_alpha(0.7)
                            pc.set_edgecolor('black')
                            pc.set_linewidth(0.5)

                        if jitter_points:
                            x_jitter = np.random.normal(x_pos, split_width * 0.08, size=len(split_data))
                            # Create darker version of violin color
                            split_color = split_colors_palette[j]
                            if isinstance(split_color, str):
                                split_color = mcolors.to_rgb(split_color)
                            darker_color = tuple(c * 0.7 for c in split_color[:3])
                            ax1.scatter(x_jitter, split_data,
                                    c=[darker_color], s=jitter_dot_size, alpha=0.8, edgecolors='none')

                        if plot_mean:
                            # Calculate mean of expressing cells for this split
                            if percent_expressing_layer is not None:
                                # Use DataFrame indexing to get split data for mean calculation
                                split_indices = group_data[group_data['split'] == split_cat].index
                                percent_split_data = percent_df.loc[split_indices, 'percent_expr']
                                expressing_mask = percent_split_data > 0

                                if expressing_mask.sum() > 0:
                                    split_mean = split_data[expressing_mask].mean()
                                else:
                                    split_mean = 0
                            else:
                                # Fallback: use expression data itself to find expressing cells
                                expressing_mask = split_data > 0
                                if expressing_mask.sum() > 0:
                                    split_mean = split_data[expressing_mask].mean()
                                else:
                                    split_mean = 0

                            mean_positions.append(x_pos)
                            mean_values.append(split_mean)
                            subplot_means.append(split_mean)

                    else:
                        _draw_empty_split_placeholder(ax1, x_pos, split_width * 0.8, split_colors_palette[j])

                if not group_has_data:
                    _draw_empty_placeholder_clean(ax1, i, "No data")

            if plot_mean:
                for pos, mean_val in zip(mean_positions, mean_values):
                    ax2.scatter(pos, mean_val, c=mean_color, s=60, marker='o',
                            edgecolors='white', linewidth=1, zorder=10)

            # Cell statistics for split violins - positioned closer to x-axis
            if show_fraction:
                for i, group in enumerate(groups):
                    group_data = subset_data[subset_data['group'] == group]

                    for j, split_cat in enumerate(global_splits):
                        split_data = group_data[group_data['split'] == split_cat]['expression'].values
                        x_pos = i - (violin_width/2) + (j + 0.5) * split_width

                        if len(split_data) > 0:
                            n_cells = len(split_data)
                            if percent_expressing_layer is not None:
                                # Use DataFrame indexing to get split data for fraction calculation
                                split_indices = group_data[group_data['split'] == split_cat].index
                                percent_split_data = percent_df.loc[split_indices, 'percent_expr']
                                frac_expressing = (percent_split_data > 0).mean()
                            else:
                                frac_expressing = (split_data > 0).mean()

                            # Cell count - positioned just below x-axis
                            ax1.text(x_pos, -0.08, f'{n_cells}', ha='center', va='top',
                                    fontsize=number_fontsize, weight='bold',
                                    transform=ax1.get_xaxis_transform())
                            # Fraction expressing - below cell count
                            ax1.text(x_pos, -0.16, f'{frac_expressing:.{number_decimal_places}f}',
                                    ha='center', va='top', fontsize=number_fontsize-1,
                                    transform=ax1.get_xaxis_transform())
                        else:
                            # Empty split indicators
                            ax1.text(x_pos, -0.08, '0', ha='center', va='top',
                                    fontsize=number_fontsize, weight='bold', color='gray',
                                    transform=ax1.get_xaxis_transform())
                            ax1.text(x_pos, -0.16, '--', ha='center', va='top',
                                    fontsize=number_fontsize-1, color='gray',
                                    transform=ax1.get_xaxis_transform())

            # Legend only on first subplot (with custom split order)
            if idx == 0 and show_legend:
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=split_colors_palette[j], alpha=0.7, label=split_cat)
                                for j, split_cat in enumerate(global_splits)]
                ax1.legend(handles=legend_elements, loc='upper right', fontsize=legend_fontsize)

        # Store data for scaling
        subplot_y_data[idx] = subplot_expressions
        subplot_mean_data[idx] = [m for m in subplot_means if m > 0]
        all_means.extend([m for m in subplot_means if m > 0])

        # Proper free_y implementation
        if free_y and len(subplot_expressions) > 0:
            # Independent y-axis scaling for this subplot
            y_max_local = max(subplot_expressions)
            y_min_local = max(0, min(subplot_expressions))
            y_range_local = y_max_local - y_min_local
            y_limit_min_local = max(0, y_min_local - (y_range_local * 0.05))
            y_limit_max_local = y_max_local + (y_range_local * 0.1)
            ax1.set_ylim(y_limit_min_local, y_limit_max_local)
        elif len(subplot_expressions) > 0:
            # Use global y-axis scaling (free_y=False)
            ax1.set_ylim(global_y_limit_min, global_y_limit_max)
        else:
            # No data in this subplot, use global limits as fallback
            ax1.set_ylim(global_y_limit_min, global_y_limit_max)

        # Format x-axis with customizable labels (now in custom order)
        ax1.set_xticks(range(len(groups)))
        ax1.set_xticklabels(groups, rotation=xlabel_rotation, ha=xlabel_ha, fontsize=xlabel_fontsize)

        # Position group labels below stats (now in custom order)
        for i, group in enumerate(groups):
            # Group labels positioned below cell stats
            ax1.text(i, -0.24, group, ha=xlabel_ha, va='top',
                    fontsize=xlabel_fontsize, rotation=xlabel_rotation,
                    transform=ax1.get_xaxis_transform())

        # Hide the default x-axis labels since we're drawing custom ones
        ax1.set_xticklabels([])

        if is_faceted:
            if col == 0:
                y_label = f'{gene} Expression'
                if layer is not None:
                    y_label += f' ({layer})'
                ax1.set_ylabel(y_label, fontsize=ylabel_fontsize, color='black')

            if plot_mean:
                if col == n_cols - 1:
                    ax2.set_ylabel('Mean Expression', fontsize=ylabel_mean_fontsize, color=mean_color)
                ax2.tick_params(axis='y', labelcolor=mean_color, labelsize=axis_tick_fontsize)
            else:
                ax2.set_yticks([])

            # Subplot title
            subplot_title = ""
            if facet_col_cat is not None and facet_cat is not None:
                subplot_title = f"{facet_col_cat} - {facet_cat}"
            elif facet_col_cat is not None:
                subplot_title = f"{facet_col_cat}"
            elif facet_cat is not None:
                subplot_title = f"{facet_cat}"

            if subplot_title:
                ax1.set_title(subplot_title, fontsize=subtitle_fontsize, pad=2)
        else:
            # Single plot labels
            y_label = f'{gene} Expression'
            if layer is not None:
                y_label += f' ({layer})'
            ax1.set_ylabel(y_label, fontsize=ylabel_fontsize, color='black')

            if plot_mean:
                ax2.set_ylabel('Mean Expression', fontsize=ylabel_mean_fontsize, color=mean_color)
                ax2.tick_params(axis='y', labelcolor=mean_color)

        ax1.tick_params(axis='y', labelsize=axis_tick_fontsize)
        ax1.tick_params(axis='x', labelsize=axis_tick_fontsize)
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    # Calculate mean-axis scaling using expressing-cells-only means
    if plot_mean:
        if len(all_means) > 0:
            global_max_mean = max(all_means)
            global_max_expression = plot_data['expression'].max()

            if global_max_mean > 0:
                mean_axis_scale_factor = global_max_expression / global_max_mean
                print(f"📏 Mean-axis scaling factor: {mean_axis_scale_factor:.3f}")
            else:
                mean_axis_scale_factor = 1.0
                print(f"⚠️  No positive means found, using scale factor: 1.0")
        else:
            mean_axis_scale_factor = 1.0
            print(f"⚠️  No means collected, using scale factor: 1.0")

    # Apply ylim to all subplots if specified (overrides free_y behavior)
    if ylim is not None:
        for idx, (ax1, ax2, row, col) in enumerate(ax_pairs):
            ax1.set_ylim(ylim)  # Only affects left axis (violin data)

    # Proper free_mean_y implementation
    if plot_mean:
        if free_mean_y:
            # Independent scaling for each subplot's mean axis
            print(f"🎯 Free mean y-axis mode: independent scaling per subplot")
            for idx, (_, ax2, _, _) in enumerate(ax_pairs):
                subplot_expressions_local = subplot_y_data[idx]
                subplot_means_local = subplot_mean_data[idx]

                if len(subplot_expressions_local) > 0 and len(subplot_means_local) > 0:
                    max_expr_local = max(subplot_expressions_local)
                    max_mean_local = max(subplot_means_local)
                    if max_mean_local > 0:
                        local_scale_factor = max_expr_local / max_mean_local
                        local_mean_axis_max = max_expr_local / local_scale_factor * 1.1
                        ax2.set_ylim(0, local_mean_axis_max)
                    else:
                        ax2.set_ylim(0, 1)
                else:
                    ax2.set_ylim(0, 1)
        else:
            # Global scaling for all mean axes
            if mean_axis_scale_factor > 0:
                mean_axis_max = global_max_expression / mean_axis_scale_factor * 1.1
                for _, ax2, _, _ in ax_pairs:
                    ax2.set_ylim(0, mean_axis_max)
                print(f"🎯 Global mean expression axis max: {mean_axis_max:.3f} (mean-axis scaling)")
            else:
                for _, ax2, _, _ in ax_pairs:
                    ax2.set_ylim(0, 1)
                print(f"🎯 Fallback mean expression axis max: 1.0")

    # Final layout and titles
    main_title = title or f'{gene} Expression'
    if split_by is not None:
        main_title += f' (split by {split_by})'
    if layer is not None:
        main_title += f' [{layer}]'

    if free_y or free_mean_y:
        scale_info = []
        if free_y:
            scale_info.append("free gene expr.")
        if free_mean_y:
            scale_info.append("free mean expr.")
        main_title += f' ({", ".join(scale_info)})'

    fig.suptitle(main_title, fontsize=title_fontsize, y=0.98)

    # Layout adjustment
    if show_fraction:
        plt.subplots_adjust(left=0.15, bottom=0.35, top=0.92, right=0.95, hspace=0.3)
    else:
        plt.subplots_adjust(left=0.15, bottom=0.2, top=0.92, right=0.95, hspace=0.3)

    plt.tight_layout()

    return fig


def vlnplot_scvi(adata, gene, group_by,
                    title=None,
                    layer=None,
                    raw_layer='raw',
                    expression_threshold=0.1,
                    split_by=None,
                    facet_by=None,
                    facet_col=None,
                    group_order=None,
                    split_order=None,
                    facet_order=None,
                    facet_col_order=None,
                    group_colors=None,
                    split_colors=None,
                    jitter_points=True,
                    jitter_dot_size=12,
                    plot_mean=True,
                    show_fraction=True,
                    show_stats=True,
                    show_legend=None,            # Smart default: False for regular, True for split
                    legend_loc='upper right',    # Legend position
                    show_group_legend=None,      # Explicit group legend control
                    group_legend_loc='below',    # Group legend position
                    group_legend_fontsize=None,  # Inherits from legend_fontsize
                    show_xlabel=True,            # Control x-axis label visibility
                    group_labels=None,           # X-axis abbreviation dictionary
                    scvi_model=None,             # scVI model for differential expression
                    group_effects=None,          # List of group comparisons for DE
                    split_effects=None,          # List of split comparisons for DE
                    split_stats=None,            # Pre-computed split_by DE statistics - dict with group_by keys, each containing {'proba_de', 'lfc_mean'}
                    annotate_stats=None,         # Enable statistical annotations (auto if effects specified)
                    detailed_stats=False,        # Show detailed P(DE) labels vs clean stars
                    proba_de_thresholds=(0.6, 0.8, 0.95),  # Thresholds for *, **, ***
                    de_mode='change',            # scVI DE mode: 'change' or 'vanilla'
                    de_delta=0.25,               # Effect size threshold for DE
                    title_fontsize=14,           # Main plot title
                    subtitle_fontsize=9,         # Individual subplot titles
                    ylabel_fontsize=10,          # Y-axis labels (left side)
                    ylabel_mean_fontsize=10,     # Mean expression y-axis labels (right side)
                    xlabel_fontsize=8,           # X-axis group labels (existing, but now more explicit)
                    axis_tick_fontsize=8,        # Axis tick numbers
                    legend_fontsize=8,           # Split legend
                    number_fontsize=6,           # Cell count/fraction numbers (existing)
                    number_decimal_places=2,
                    xlabel_rotation=45,
                    xlabel_ha='right',
                    figsize=(12, 8),
                    facet_figsize=None,
                    facet_ncols=None,
                    mean_color='black',
                    free_y=True,
                    free_mean_y=False,
                    ylim=None):
    """
    Create violin plots optimized for scVI-transformed single-cell data with statistical analysis.

    This function handles the dual-data nature of scVI analysis: using scVI-transformed
    values for expression distributions while using raw counts for accurate fraction
    expressing calculations. Supports flexible statistical comparisons for any grouping variables.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing scVI-transformed data in .X and raw counts in layers
    gene : str
        Gene name to plot (must exist in adata.var_names)
    group_by : str
        Column name in adata.obs for grouping cells on x-axis
        (e.g., 'condition', 'celltype', 'timepoint', 'treatment')
    split_by : str, optional
        Column name in adata.obs for split violin plots
        (e.g., 'treatment', 'genotype', 'stimulus', 'batch')
    raw_layer : str, default 'raw'
        Layer containing raw counts for fraction expressing calculation.
        Falls back to main data with threshold if layer not found.
    expression_threshold : float, default 0.1
        Threshold for determining expressing cells when using scVI data directly

    Statistical Analysis Parameters
    ------------------------------
    scvi_model : scvi model, optional
        Trained scVI model for differential expression analysis
    group_effects : list of tuples, optional
        Group comparisons for differential expression, e.g. [('condition1', 'condition2')]
        Compares between categories of group_by variable
    split_effects : list of tuples, optional
        Split comparisons for differential expression, e.g. [('treatment1', 'treatment2')]
        Compares between categories of split_by variable within each group
    split_stats : dict, optional
        Pre-computed split_by DE statistics. Dict with group_by categories as keys,
        each containing {'proba_de': float, 'lfc_mean': float}.
        Use compute_split_effects_within_groups() to generate this.
    annotate_stats : bool, optional
        Enable statistical annotations. Auto-enabled if any effects or split_stats provided.
    detailed_stats : bool, default False
        Show detailed statistics (P-values/Bayes factors + LFC) vs clean stars only
    proba_de_thresholds : tuple, default (0.6, 0.8, 0.95)
        Probability thresholds for *, **, *** significance levels
    de_mode : str, default 'change'
        scVI differential expression mode: 'change' or 'vanilla'.
        'change' mode focuses on meaningful effect sizes and is generally recommended.
    de_delta : float, default 0.25
        Effect size threshold for differential expression

    Styling Parameters
    ------------------
    title : str, optional
        Main plot title. If None, uses gene name
    group_colors : dict, optional
        Colors for group_by categories, e.g. {'condition1': 'red', 'condition2': 'blue'}
    split_colors : dict, optional
        Colors for split_by categories, e.g. {'treatment1': 'green', 'treatment2': 'orange'}
    group_labels : dict, optional
        Abbreviated labels for x-axis, e.g. {'Nonlesional': 'NL', 'SADBE': 'S'}
    figsize : tuple, default (12, 8)
        Figure size in inches
    jitter_points : bool, default True
        Show individual data points with jitter
    show_fraction : bool, default True
        Show cell counts and fraction expressing below plot
    show_legend : bool, optional
        Show legend. Auto-determined based on split_by if not specified
    xlabel_rotation : float, default 45
        Rotation angle for x-axis labels
    *_fontsize : int
        Font sizes for various plot elements (title, labels, legend, etc.)

    Returns
    -------
    matplotlib.figure.Figure
        The generated violin plot figure

    Examples
    --------
    Pattern A: Split effects within groups (e.g., treatment effects within conditions)

    >>> # Pre-compute treatment effects within each condition
    >>> treatment_by_condition = ssc.compute_split_effects_within_groups(
    ...     adata, 'GNLY', 'condition', 'treatment', ['dupi', 'pre'], scvi_model
    ... )
    >>> # Plot with statistical annotations
    >>> ssc.vlnplot_scvi(
    ...     adata, 'GNLY', group_by='condition', split_by='treatment',
    ...     split_effects=[('dupi', 'pre')], split_stats=treatment_by_condition,
    ...     detailed_stats=True
    ... )

    Pattern B1: Pure group comparisons (subset first)

    >>> # Compare conditions within dupilumab-treated cells only
    >>> adata_dupi = adata[adata.obs['treatment'] == 'dupi'].copy()
    >>> ssc.vlnplot_scvi(
    ...     adata_dupi, 'GNLY', group_by='condition',
    ...     scvi_model=scvi_model, group_effects=[('Nonlesional', 'SADBE')]
    ... )

    Pattern B3: Custom cross-group-treatment comparisons

    >>> # Create combined grouping for flexible comparisons
    >>> adata.obs['condition_treatment'] = adata.obs['condition'] + '_' + adata.obs['treatment']
    >>> ssc.vlnplot_scvi(
    ...     adata, 'GNLY', group_by='condition_treatment',
    ...     scvi_model=scvi_model,
    ...     group_effects=[('Nonlesional_dupi', 'SADBE_dupi'),  # Treatment responses
    ...                    ('Nonlesional_pre', 'SADBE_pre')]    # Baseline differences
    ... )

    Notes
    -----
    - For best performance, pre-compute split statistics using compute_split_effects_within_groups()
    - Statistical comparisons require scvi_model unless pre-computed split_stats provided
    - Use detailed_stats=True to see P-values and log fold changes on plot
    - Function automatically handles scVI's 1-vs-all DE results and converts to pairwise comparisons
    """

    import warnings

    # Input validation
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")
    if group_by not in adata.obs.columns:
        raise ValueError(f"Group column '{group_by}' not found in adata.obs")

    # Statistical framework setup
    if annotate_stats is None:
        # Auto-enable if any effects are specified or split stats provided
        annotate_stats = bool(group_effects or split_effects or split_stats)

    if annotate_stats and not scvi_model and not split_stats:
        raise ValueError("scvi_model is required when annotate_stats=True or when group_effects/split_effects are specified (unless split_stats is provided)")

    # Validate statistical effects
    if group_effects and not isinstance(group_effects, list):
        raise ValueError("group_effects must be a list of tuples, e.g. [('group1', 'group2')]")
    if split_effects and not isinstance(split_effects, list):
        raise ValueError("split_effects must be a list of tuples, e.g. [('condition1', 'condition2')]")

    if split_effects and not split_by:
        raise ValueError("split_by must be specified when using split_effects")

    # Get expression data (scVI-transformed)
    if layer is not None:
        if layer not in adata.layers.keys():
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        expression_data = adata[:, gene].layers[layer].toarray().flatten()
        data_source = f"layer '{layer}'"
    else:
        expression_data = adata[:, gene].X.toarray().flatten()
        data_source = "adata.X (scVI-transformed)"

    # Handle fraction expressing data with fallback logic
    if show_fraction:
        if raw_layer in adata.layers:
            # Use raw layer for accurate fraction calculation
            fraction_data = adata[:, gene].layers[raw_layer].toarray().flatten()
            fraction_source = f"layer '{raw_layer}'"
            fraction_threshold = 0
            print(f"📊 Using {raw_layer} layer for fraction expressing calculation")
        else:
            # Fallback to scVI data with threshold
            warnings.warn(f"Layer '{raw_layer}' not found. Using scVI data with threshold={expression_threshold} for fraction calculation.")
            fraction_data = expression_data
            fraction_source = f"{data_source} with threshold"
            fraction_threshold = expression_threshold
            print(f"⚠️  Fallback: Using scVI data with threshold {expression_threshold} for fractions")
    else:
        fraction_data = None
        fraction_source = "not used"
        fraction_threshold = 0

    print(f"📊 Expression data from: {data_source}")
    print(f"📊 Fraction data from: {fraction_source}")

    # Create DataFrames for clean indexing
    plot_data = pd.DataFrame({
        'group': adata.obs[group_by].values,
        'expression': expression_data
    }, index=adata.obs.index)

    # Add split data if specified
    if split_by is not None:
        if split_by not in adata.obs.columns:
            raise ValueError(f"Split column '{split_by}' not found in adata.obs")
        plot_data['split'] = adata.obs[split_by].values
        print(f"📊 Split by: {split_by}")
    else:
        plot_data['split'] = None

    if show_fraction:
        fraction_df = pd.DataFrame({
            'fraction_expr': fraction_data
        }, index=adata.obs.index)
    else:
        fraction_df = None

    # Get unique groups and order them
    if group_order is not None:
        groups = [g for g in group_order if g in plot_data['group'].unique()]
        # Add any missing groups
        missing_groups = [g for g in plot_data['group'].unique() if g not in groups]
        groups.extend(missing_groups)
    else:
        groups = sorted(plot_data['group'].unique())

    print(f"📋 Groups: {groups}")

    # Get unique splits if splitting
    if split_by is not None:
        if split_order is not None:
            splits = [s for s in split_order if s in plot_data['split'].unique()]
            # Add any missing splits
            missing_splits = [s for s in plot_data['split'].unique() if s not in splits]
            splits.extend(missing_splits)
        else:
            splits = sorted(plot_data['split'].unique())
        print(f"📋 Splits: {splits}")
    else:
        splits = [None]

    # Compute statistical effects if requested
    gene_statistics = {}
    if annotate_stats:
        print(f"📊 Computing statistical effects for {gene}...")
        gene_statistics = _process_statistical_effects(
            adata, gene, scvi_model, group_effects, split_effects,
            group_by, split_by, de_mode, de_delta, groups, splits
        )
        print(f"📊 Computed {len(gene_statistics)} statistical comparisons")

    # Smart default legend behavior
    if show_legend is None:
        if split_by is not None:
            show_legend = True   # Default to True for split mode
        else:
            show_legend = False  # Default to False for regular mode

    # Set default group legend fontsize
    if group_legend_fontsize is None:
        group_legend_fontsize = legend_fontsize

    # Handle group colors
    if group_colors is not None:
        # Use provided group colors with validation
        group_colors_list = _get_colors_from_dict(group_colors, groups,
                                                  sns.color_palette("tab10", len(groups)), "Group colors")
    else:
        # Use matplotlib/seaborn default colors
        group_colors_list = sns.color_palette("tab10", len(groups))

    print(f"🎨 Group colors: {[f'{group}={color}' for group, color in zip(groups, group_colors_list)]}")

    # Create simple plot for now (no splits or facets yet)
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()  # Secondary axis for means

    # Create violin plots (with or without splits)
    if split_by is None:
        # Original single violin logic
        for i, group in enumerate(groups):
            group_data = plot_data[plot_data['group'] == group]['expression']

            if len(group_data) > 0:
                # Create violin plot with specified colors
                violin_parts = ax.violinplot([group_data], positions=[i], widths=0.6, showmeans=False, showextrema=False)

                # Apply custom colors to violin
                group_color = group_colors_list[i]
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(group_color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)

                # Add jitter points for regular violins
                if jitter_points:
                    x_jitter = np.random.normal(i, 0.1, size=len(group_data))
                    # Create darker version of violin color
                    if isinstance(group_color, str):
                        group_color_rgb = mcolors.to_rgb(group_color)
                    else:
                        group_color_rgb = group_color[:3]
                    darker_color = tuple(c * 0.7 for c in group_color_rgb)
                    ax.scatter(x_jitter, group_data,
                            c=[darker_color], s=jitter_dot_size, alpha=0.8, edgecolors='none')

                # Add mean expression points if enabled
                if plot_mean:
                    if fraction_df is not None:
                        # Use raw data for identifying expressing cells for mean calculation
                        group_indices = plot_data[plot_data['group'] == group].index
                        group_fraction_data = fraction_df.loc[group_indices, 'fraction_expr']
                        expressing_mask = group_fraction_data > fraction_threshold
                        if expressing_mask.sum() > 0:
                            group_mean = group_data[expressing_mask].mean()
                        else:
                            group_mean = 0
                    else:
                        # Fallback: use expression data itself
                        expressing_mask = group_data > expression_threshold
                        if expressing_mask.sum() > 0:
                            group_mean = group_data[expressing_mask].mean()
                        else:
                            group_mean = 0

                    if ax2 is not None:
                        ax2.scatter(i, group_mean, c=mean_color, s=60, marker='o',
                                edgecolors='white', linewidth=1, zorder=10)

    # Add fraction numbers for regular violins
    if show_fraction and split_by is None:
        for i, group in enumerate(groups):
            group_data = plot_data[plot_data['group'] == group]['expression']

            if len(group_data) > 0:
                n_cells = len(group_data)
                if fraction_df is not None:
                    # Use raw data for fraction calculation
                    group_indices = plot_data[plot_data['group'] == group].index
                    group_fraction_data = fraction_df.loc[group_indices, 'fraction_expr']
                    n_expressing = (group_fraction_data > fraction_threshold).sum()
                    frac_expressing = n_expressing / n_cells if n_cells > 0 else 0
                else:
                    # Fallback: use expression data itself
                    n_expressing = (group_data > expression_threshold).sum()
                    frac_expressing = n_expressing / n_cells if n_cells > 0 else 0

                # Total cells - positioned just below x-axis
                ax.text(i, -0.08, f'{n_cells}', ha='center', va='top',
                        fontsize=number_fontsize, weight='bold',
                        transform=ax.get_xaxis_transform())
                # Expressing cells - middle row
                ax.text(i, -0.12, f'{n_expressing}', ha='center', va='top',
                        fontsize=number_fontsize-1,
                        transform=ax.get_xaxis_transform())
                # Fraction expressing - bottom row
                ax.text(i, -0.16, f'{frac_expressing:.{number_decimal_places}f}',
                        ha='center', va='top', fontsize=number_fontsize-1,
                        transform=ax.get_xaxis_transform())
            else:
                # Empty data indicators
                ax.text(i, -0.08, '0', ha='center', va='top',
                        fontsize=number_fontsize, weight='bold', color='gray',
                        transform=ax.get_xaxis_transform())
                ax.text(i, -0.12, '0', ha='center', va='top',
                        fontsize=number_fontsize-1, color='gray',
                        transform=ax.get_xaxis_transform())
                ax.text(i, -0.16, '--', ha='center', va='top',
                        fontsize=number_fontsize-1, color='gray',
                        transform=ax.get_xaxis_transform())

    else:
        # Split violin logic
        violin_width = 0.8 / len(splits)  # Divide space among splits

        for i, group in enumerate(groups):
            group_data_all = plot_data[plot_data['group'] == group]

            for j, split in enumerate(splits):
                # Get data for this group + split combination
                split_data = group_data_all[group_data_all['split'] == split]['expression']

                if len(split_data) > 0:
                    # Calculate position for this split within the group
                    x_offset = (j - (len(splits) - 1) / 2) * violin_width
                    x_pos = i + x_offset

                    # Create violin plot
                    violin_parts = ax.violinplot([split_data], positions=[x_pos], widths=violin_width * 0.8,
                                                showmeans=False, showextrema=False)

                    # Apply split colors
                    if split_colors and split in split_colors:
                        split_color = split_colors[split]
                    else:
                        # Default colors if no custom split colors provided
                        default_colors = ['purple', 'orange', 'green', 'red', 'blue']
                        split_color = default_colors[j % len(default_colors)]

                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(split_color)
                        pc.set_alpha(0.7)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(0.5)

                    # Add jitter points
                    if jitter_points:
                        x_jitter = np.random.normal(x_pos, violin_width * 0.08, size=len(split_data))
                        # Create darker version of violin color
                        if isinstance(split_color, str):
                            split_color_rgb = mcolors.to_rgb(split_color)
                        else:
                            split_color_rgb = split_color[:3]
                        darker_color = tuple(c * 0.7 for c in split_color_rgb)
                        ax.scatter(x_jitter, split_data,
                                c=[darker_color], s=jitter_dot_size, alpha=0.8, edgecolors='none')

                    # Add mean expression points if enabled
                    if plot_mean:
                        if fraction_df is not None:
                            # Use raw data for identifying expressing cells for mean calculation
                            split_indices = group_data_all[group_data_all['split'] == split].index
                            split_fraction_data = fraction_df.loc[split_indices, 'fraction_expr']
                            expressing_mask = split_fraction_data > fraction_threshold
                            if expressing_mask.sum() > 0:
                                split_mean = split_data[expressing_mask].mean()
                            else:
                                split_mean = 0
                        else:
                            # Fallback: use expression data itself
                            expressing_mask = split_data > expression_threshold
                            if expressing_mask.sum() > 0:
                                split_mean = split_data[expressing_mask].mean()
                            else:
                                split_mean = 0

                        if ax2 is not None:
                            ax2.scatter(x_pos, split_mean, c=mean_color, s=60, marker='o',
                                    edgecolors='white', linewidth=1, zorder=10)

    # Add fraction numbers for split violins
    if show_fraction and split_by is not None:
        for i, group in enumerate(groups):
            group_data_all = plot_data[plot_data['group'] == group]

            for j, split in enumerate(splits):
                split_data = group_data_all[group_data_all['split'] == split]['expression']
                x_offset = (j - (len(splits) - 1) / 2) * violin_width
                x_pos = i + x_offset

                if len(split_data) > 0:
                    n_cells = len(split_data)
                    if fraction_df is not None:
                        # Use raw data for fraction calculation
                        split_indices = group_data_all[group_data_all['split'] == split].index
                        split_fraction_data = fraction_df.loc[split_indices, 'fraction_expr']
                        n_expressing = (split_fraction_data > fraction_threshold).sum()
                        frac_expressing = n_expressing / n_cells if n_cells > 0 else 0
                    else:
                        # Fallback: use expression data itself
                        n_expressing = (split_data > expression_threshold).sum()
                        frac_expressing = n_expressing / n_cells if n_cells > 0 else 0

                    # Total cells - positioned just below x-axis
                    ax.text(x_pos, -0.08, f'{n_cells}', ha='center', va='top',
                            fontsize=number_fontsize, weight='bold',
                            transform=ax.get_xaxis_transform())
                    # Expressing cells - middle row
                    ax.text(x_pos, -0.12, f'{n_expressing}', ha='center', va='top',
                            fontsize=number_fontsize-1,
                            transform=ax.get_xaxis_transform())
                    # Fraction expressing - bottom row
                    ax.text(x_pos, -0.16, f'{frac_expressing:.{number_decimal_places}f}',
                            ha='center', va='top', fontsize=number_fontsize-1,
                            transform=ax.get_xaxis_transform())
                else:
                    # Empty data indicators
                    ax.text(x_pos, -0.08, '0', ha='center', va='top',
                            fontsize=number_fontsize, weight='bold', color='gray',
                            transform=ax.get_xaxis_transform())
                    ax.text(x_pos, -0.12, '0', ha='center', va='top',
                            fontsize=number_fontsize-1, color='gray',
                            transform=ax.get_xaxis_transform())
                    ax.text(x_pos, -0.16, '--', ha='center', va='top',
                            fontsize=number_fontsize-1, color='gray',
                            transform=ax.get_xaxis_transform())

    # Legend creation logic
    legends_created = []

    # Determine which legends to show
    show_split_legend = show_legend and split_by is not None
    show_group_legend_flag = show_group_legend or (show_legend and split_by is None)

    # Create split legend if requested
    if show_split_legend:
        # Create legend elements for splits
        split_legend_elements = []
        for j, split in enumerate(splits):
            if split_colors and split in split_colors:
                split_color = split_colors[split]
            else:
                # Use same default colors as in the plotting
                default_colors = ['purple', 'orange', 'green', 'red', 'blue']
                split_color = default_colors[j % len(default_colors)]

            split_legend_elements.append(plt.Rectangle((0,0), 1, 1,
                                                     facecolor=split_color, alpha=0.7,
                                                     edgecolor='black', linewidth=0.5,
                                                     label=split))

        # Handle custom legend positions for split legend
        if legend_loc == 'below':
            split_legend = ax.legend(handles=split_legend_elements, fontsize=legend_fontsize,
                                   title=split_by, title_fontsize=legend_fontsize,
                                   bbox_to_anchor=(0.5, -0.25), loc='center',
                                   ncol=len(splits))
        elif legend_loc == 'right':
            split_legend = ax.legend(handles=split_legend_elements, fontsize=legend_fontsize,
                                   title=split_by, title_fontsize=legend_fontsize,
                                   bbox_to_anchor=(1.15, 0.5), loc='center left')
        else:
            split_legend = ax.legend(handles=split_legend_elements, loc=legend_loc, fontsize=legend_fontsize,
                                   title=split_by, title_fontsize=legend_fontsize)
        legends_created.append(split_legend)

    # Create group legend if requested
    if show_group_legend_flag:
        # Create legend elements for groups
        group_legend_elements = []
        for i, group in enumerate(groups):
            group_color = group_colors_list[i]
            group_legend_elements.append(plt.Rectangle((0,0), 1, 1,
                                                      facecolor=group_color, alpha=0.7,
                                                      edgecolor='black', linewidth=0.5,
                                                      label=group))

        # Position group legend
        if group_legend_loc == 'below':
            # Adjust position if split legend is also below
            y_offset = -0.25 if not (show_split_legend and legend_loc == 'below') else -0.35
            group_legend = ax.legend(handles=group_legend_elements, fontsize=group_legend_fontsize,
                                   title=group_by, title_fontsize=group_legend_fontsize,
                                   bbox_to_anchor=(0.5, y_offset), loc='center',
                                   ncol=len(groups))
        elif group_legend_loc == 'right':
            # Adjust position if split legend is also right
            x_offset = 1.15 if not (show_split_legend and legend_loc == 'right') else 1.30
            group_legend = ax.legend(handles=group_legend_elements, fontsize=group_legend_fontsize,
                                   title=group_by, title_fontsize=group_legend_fontsize,
                                   bbox_to_anchor=(x_offset, 0.5), loc='center left')
        else:
            group_legend = ax.legend(handles=group_legend_elements, loc=group_legend_loc,
                                   fontsize=group_legend_fontsize,
                                   title=group_by, title_fontsize=group_legend_fontsize)

        # If we have both legends, we need to add the group legend manually
        if show_split_legend:
            ax.add_artist(split_legend)  # Keep the split legend

    # Format axes
    ax.set_xticks(range(len(groups)))

    # Determine x-axis labels (use abbreviations if provided and group legend is shown)
    if group_labels is not None and show_group_legend_flag:
        # Use abbreviated labels when group legend is shown
        x_labels = [group_labels.get(group, group) for group in groups]
        print(f"📋 Using abbreviated x-axis labels: {dict(zip(groups, x_labels))}")
    else:
        # Use full group names
        x_labels = groups

    # Position x-axis labels based on whether fraction numbers are shown and xlabel visibility
    if show_xlabel:
        if show_fraction:
            # Move labels below the three-row numbers
            ax.set_xticklabels(x_labels, rotation=xlabel_rotation, ha=xlabel_ha, fontsize=xlabel_fontsize)
            # Move x-axis labels below numbers with appropriate spacing
            ax.tick_params(axis='x', which='major', pad=70)  # Reduced padding to bring labels up a bit
        else:
            # Normal position when no numbers shown
            ax.set_xticklabels(x_labels, rotation=xlabel_rotation, ha=xlabel_ha, fontsize=xlabel_fontsize)
    else:
        # Hide x-axis labels
        ax.set_xticklabels([])

    ax.set_ylabel(f'{gene} Expression', fontsize=ylabel_fontsize)

    if plot_mean:
        ax2.set_ylabel('Mean Expression', fontsize=ylabel_mean_fontsize)

    # Set title
    plot_title = title or f'{gene} Expression (scVI)'
    ax.set_title(plot_title, fontsize=title_fontsize)

    # Add statistical annotations
    if annotate_stats and gene_statistics:
        print(f"📊 Adding statistical annotations...")
        y_min, y_max = ax.get_ylim()

        # Calculate annotation heights
        annotation_height_start = y_max * 1.05
        annotation_height_step = y_max * 0.08

        annotation_count = 0

        # Add group effect annotations
        if group_effects:
            for comparison in group_effects:
                if isinstance(comparison, tuple) and len(comparison) == 2:
                    group1, group2 = comparison
                    stats_key = f"group_{group1}_vs_{group2}"

                    if stats_key in gene_statistics:
                        try:
                            x1 = groups.index(group1)
                            x2 = groups.index(group2)
                            y_pos = annotation_height_start + annotation_count * annotation_height_step

                            _add_statistical_annotation(
                                ax, x1, x2, y_pos, gene_statistics[stats_key],
                                detailed_stats, proba_de_thresholds, de_mode
                            )
                            annotation_count += 1
                        except ValueError:
                            print(f"⚠️ Groups {group1} or {group2} not found in plot")

        # Add split effect annotations - group-specific approach using pre-computed stats
        if split_effects and split_by and splits and len(splits) == 2 and split_stats:
            split1, split2 = splits[0], splits[1]

            print(f"📊 Adding group-specific split annotations: {split1} vs {split2}")

            # Add split comparison annotations for each group individually
            for i, group in enumerate(groups):
                if group in split_stats:
                    # Extract pre-computed statistics for this specific group
                    group_stats = split_stats[group]
                    proba_de = group_stats.get('proba_de', 0)
                    lfc = group_stats.get('lfc_mean', 0)

                    # Convert Series to scalar if needed
                    if hasattr(proba_de, 'item'):
                        proba_de = proba_de.item()
                    if hasattr(lfc, 'item'):
                        lfc = lfc.item()

                    # Determine significance level for this group
                    if proba_de >= proba_de_thresholds[2]:  # 95%
                        stars = "***"
                    elif proba_de >= proba_de_thresholds[1]:  # 80%
                        stars = "**"
                    elif proba_de >= proba_de_thresholds[0]:  # 60%
                        stars = "*"
                    else:
                        stars = "ns"

                    # Only add annotations if significant for this group
                    if stars != "ns":
                        print(f"   ✅ {group}: {split1} vs {split2} = {stars} (P={proba_de:.3f}, LFC={lfc:.2f})")

                        # Calculate x positions for the two split categories within this group
                        x1 = i - 0.2  # Position of first split category
                        x2 = i + 0.2  # Position of second split category

                        # Use a lower annotation height for split comparisons
                        y_pos = y_max * 0.95

                        # Add significance line
                        ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1)
                        ax.plot([x1, x1], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)
                        ax.plot([x2, x2], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)

                        # Add annotation text (stars or detailed stats)
                        x_center = (x1 + x2) / 2
                        if detailed_stats:
                            annotation = f"{stars}\nP={proba_de:.2f}"
                            if abs(lfc) >= 0.1:  # Only show LFC if meaningful
                                annotation += f"\nLFC={lfc:.1f}"
                        else:
                            annotation = stars

                        ax.text(x_center, y_pos + y_max*0.02, annotation, ha='center', va='bottom',
                               fontsize=14, fontweight='bold')
                    else:
                        print(f"   ○ {group}: {split1} vs {split2} = ns (P={proba_de:.3f})")
                else:
                    print(f"   ⚠️ {group}: No split statistics provided")

        # Adjust y-axis to accommodate annotations
        if annotation_count > 0:
            new_y_max = annotation_height_start + annotation_count * annotation_height_step + y_max * 0.1
            ax.set_ylim(0, new_y_max)
        else:
            ax.set_ylim(0, y_max)
    else:
        # Ensure y-axis starts at 0 to show full violin range
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max)

    # Adjust layout for three rows of numbers
    if show_fraction:
        plt.subplots_adjust(bottom=0.35)

    plt.tight_layout()

    print(f"✅ Basic vlnplot_scvi completed for {gene}")
    print(f"📊 Shows: Total cells | Expressing cells | Fraction expressing")
    return fig