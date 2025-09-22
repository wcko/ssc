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


def _process_comparisons(adata, gene, scvi_model, comparisons, group_by, split_by=None):
    """Simple function to handle comparisons list and return stats for plotting."""
    if not comparisons or not scvi_model:
        return {}

    print(f"🔄 Processing {len(comparisons)} comparisons for {gene}...")
    stats = {}

    for i, comparison in enumerate(comparisons):
        if len(comparison) not in [3, 4]:
            print(f"⚠️ Skipping invalid comparison {i}: expected 3 or 4 elements, got {len(comparison)}")
            continue

        comp_type = comparison[0]

        # Parse based on tuple length
        if len(comparison) == 3:
            # 3-tuple: ('group', 'A', 'B') or old ('split', 'A', 'B')
            _, group1, group2 = comparison
            within_group = None
        else:
            # 4-tuple: ('split', 'GroupName', 'Split1', 'Split2')
            _, within_group, group1, group2 = comparison

        if comp_type == 'group':
            print(f"   📊 Group comparison: {group1} vs {group2}")

            # Use existing working DE function
            de_results = _get_or_compute_de_working(
                scvi_model, adata,
                groupby=group_by,
                group1=group1,      # Add specific group names for proper caching
                group2=group2,      # Add specific group names for proper caching
                subset_data=adata,  # Use full data for group comparisons
                mode='change'       # Use change mode (working approach)
            )

            if de_results is not None:
                # Extract stats for this specific gene and comparison
                gene_stats = _extract_pairwise_comparison(de_results, gene, group1, group2)
                if gene_stats is not None and not gene_stats.empty:
                    stats[f"group_{group1}_vs_{group2}"] = gene_stats
                    proba_de = gene_stats.get('proba_de', 0)
                    lfc_mean = gene_stats.get('lfc_mean', 0)
                    bayes_factor = gene_stats.get('bayes_factor', 0)
                    print(f"   ✅ proba_de={proba_de:.3f}, LFC={lfc_mean:.3f}, BF={bayes_factor:.1f}")
                else:
                    print(f"   ❌ Failed to extract stats for {gene}")
            else:
                print(f"   ❌ DE computation failed")

        elif comp_type == 'split':
            # Validate 4-tuple format and split_by parameter
            if len(comparison) != 4:
                print(f"   ⚠️ 'split' comparisons require 4-tuple format: ('split', 'GroupName', 'Split1', 'Split2')")
                continue
            if not split_by:
                print(f"   ⚠️ 'split' comparisons require split_by parameter")
                continue

            print(f"   📊 Split comparison: {group1} vs {group2} within {within_group}")

            # Create subset: specific group AND specific split categories
            group_mask = adata.obs[group_by] == within_group
            split_mask = adata.obs[split_by].isin([group1, group2])
            combined_mask = group_mask & split_mask
            subset_data = adata[combined_mask].copy()

            if subset_data.n_obs == 0:
                print(f"   ⚠️ No cells found for {group1}/{group2} splits within {within_group}")
                continue

            print(f"   📊 Using {subset_data.n_obs} cells from {within_group} group")

            # Use split_by as the grouping variable for DE computation
            de_results = _get_or_compute_de_working(
                scvi_model, subset_data,
                groupby=split_by,
                group1=group1,
                group2=group2,
                subset_data=subset_data,
                mode='change'
            )

            if de_results is not None:
                # Extract stats for this specific gene and comparison
                gene_stats = _extract_pairwise_comparison(de_results, gene, group1, group2)
                if gene_stats is not None and not gene_stats.empty:
                    stats[f"split_{within_group}_{group1}_vs_{group2}"] = gene_stats
                    proba_de = gene_stats.get('proba_de', 0)
                    lfc_mean = gene_stats.get('lfc_mean', 0)
                    bayes_factor = gene_stats.get('bayes_factor', 0)
                    print(f"   ✅ proba_de={proba_de:.3f}, LFC={lfc_mean:.3f}, BF={bayes_factor:.1f}")
                else:
                    print(f"   ❌ Failed to extract stats for {gene}")
            else:
                print(f"   ❌ Split DE computation failed")

        else:
            print(f"   ⚠️ Unsupported comparison type: {comp_type} (use 'group' or 'split')")

    return stats


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
        return "ns"


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



def _get_star_annotation_standard(p_value, thresholds=(0.05, 0.01, 0.001)):
    """Convert p-value to star annotation for standard statistical tests.

    Parameters
    ----------
    p_value : float
        P-value from statistical test (0-1 scale)
    thresholds : tuple
        P-value thresholds for significance levels (p<0.05, p<0.01, p<0.001)

    Returns
    -------
    str
        Star annotation: '***', '**', '*', or 'ns'
    """
    if p_value < thresholds[2]:  # p < 0.001
        return "***"
    elif p_value < thresholds[1]:  # p < 0.01
        return "**"
    elif p_value < thresholds[0]:  # p < 0.05
        return "*"
    else:
        return "ns"


def _compute_standard_statistics(group1_data, group2_data, method='ttest'):
    """Compute standard statistical tests between two groups.

    Parameters
    ----------
    group1_data : array-like
        Expression data for group 1
    group2_data : array-like
        Expression data for group 2
    method : str
        Statistical test method: 'ttest', 'wilcoxon', 'mannwhitney'

    Returns
    -------
    dict
        Statistical results with keys: pvalue, statistic, logfc, method
    """
    import numpy as np
    from scipy import stats

    # Convert to numpy arrays and remove NaN values
    group1 = np.array(group1_data).flatten()
    group2 = np.array(group2_data).flatten()
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]

    if len(group1) == 0 or len(group2) == 0:
        return {'pvalue': 1.0, 'statistic': 0.0, 'logfc': 0.0, 'method': method}

    # Calculate log fold change (log2 of mean ratios)
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    # Add small pseudocount to avoid log(0)
    pseudocount = 1e-6
    logfc = np.log2((mean2 + pseudocount) / (mean1 + pseudocount))

    # Perform statistical test
    try:
        if method == 'ttest':
            statistic, pvalue = stats.ttest_ind(group1, group2, equal_var=False)
        elif method == 'wilcoxon':
            # Use Mann-Whitney U test (equivalent to Wilcoxon rank-sum for independent samples)
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        elif method == 'mannwhitney':
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        else:
            raise ValueError(f"Unknown statistical method: {method}")

    except Exception as e:
        print(f"⚠️  Statistical test failed: {e}")
        return {'pvalue': 1.0, 'statistic': 0.0, 'logfc': logfc, 'method': method}

    return {
        'pvalue': float(pvalue) if not np.isnan(pvalue) else 1.0,
        'statistic': float(statistic) if not np.isnan(statistic) else 0.0,
        'logfc': float(logfc) if not np.isnan(logfc) else 0.0,
        'method': method
    }


def _format_statistical_text(stats_dict, display_options, detailed_stats=False):
    """Format statistical results for display.

    Parameters
    ----------
    stats_dict : dict
        Statistical results from _compute_standard_statistics
    display_options : dict
        Display options: show_pvalue, show_logfc, show_statistic, show_stars_only
    detailed_stats : bool
        Whether to show detailed statistics or just stars

    Returns
    -------
    str
        Formatted text for annotation
    """
    if stats_dict is None:
        return ""

    pvalue = stats_dict.get('pvalue', 1.0)
    statistic = stats_dict.get('statistic', 0.0)
    logfc = stats_dict.get('logfc', 0.0)
    method = stats_dict.get('method', 'unknown')

    # Get star annotation
    stars = _get_star_annotation_standard(pvalue)

    # If only showing stars
    if display_options.get('show_stars_only', False):
        return stars if stars != 'ns' else ('ns' if display_options.get('show_ns', False) else '')

    # Build annotation text
    parts = []

    if detailed_stats:
        # Detailed format: "t=2.45, p=0.023, logFC=1.23"
        if display_options.get('show_statistic', True):
            stat_name = 't' if method == 'ttest' else 'U'
            parts.append(f"{stat_name}={statistic:.2f}")
        if display_options.get('show_pvalue', True):
            if pvalue < 0.001:
                parts.append("p<0.001")
            else:
                parts.append(f"p={pvalue:.3f}")
        if display_options.get('show_logfc', True):
            parts.append(f"logFC={logfc:.2f}")
    else:
        # Simple format based on preferences
        if display_options.get('show_pvalue', False):
            if pvalue < 0.001:
                parts.append("p<0.001")
            elif pvalue < 0.01:
                parts.append("p<0.01")
            elif pvalue < 0.05:
                parts.append("p<0.05")
            else:
                parts.append(f"p={pvalue:.3f}" if display_options.get('show_ns', False) else "")
        else:
            # Just show stars
            if stars != 'ns' or display_options.get('show_ns', False):
                parts.append(stars)

    return ', '.join(filter(None, parts))


def _add_statistical_annotation_standard(ax, x1, x2, y_position, stats_dict,
                                        display_options=None, detailed_stats=False,
                                        p_thresholds=(0.05, 0.01, 0.001)):
    """Add standard statistical annotation between two x positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add annotation to
    x1, x2 : float
        X positions for comparison line
    y_position : float
        Y position for annotation
    stats_dict : dict
        Statistical results from _compute_standard_statistics
    display_options : dict
        Display preferences for annotation text
    detailed_stats : bool
        Whether to show detailed statistics
    p_thresholds : tuple
        P-value thresholds for significance stars
    """
    if stats_dict is None or not stats_dict:
        return

    # Default display options
    if display_options is None:
        display_options = {
            'show_pvalue': False,
            'show_logfc': False,
            'show_statistic': False,
            'show_stars_only': True,
            'show_ns': False
        }

    # Format annotation text
    annotation = _format_statistical_text(stats_dict, display_options, detailed_stats)

    if not annotation:
        return  # Nothing to show

    # Draw horizontal line with brackets
    line_height = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x1, x2], [y_position, y_position], 'k-', linewidth=1)
    ax.plot([x1, x1], [y_position - line_height/2, y_position + line_height/2], 'k-', linewidth=1)
    ax.plot([x2, x2], [y_position - line_height/2, y_position + line_height/2], 'k-', linewidth=1)

    # Add annotation text
    x_center = (x1 + x2) / 2
    ax.text(x_center, y_position + line_height, annotation, ha='center', va='bottom',
            fontsize=12, fontweight='bold')


def _validate_and_filter_comparisons(comparisons, adata, group_by, split_by=None):
    """
    Check which comparisons are valid given available data.
    Return valid comparisons + warning messages for invalid ones.

    Parameters
    ----------
    comparisons : list
        List of comparison tuples like ('group', 'A', 'B') or ('split', 'GroupX', 'Split1', 'Split2')
    adata : AnnData
        Annotated data object
    group_by : str
        Column name for grouping
    split_by : str, optional
        Name of split column if split comparisons are used

    Returns
    -------
    valid_comparisons : list
        List of comparisons that have sufficient data
    warnings : list
        List of warning messages for skipped comparisons
    """
    if comparisons is None:
        return [], []

    valid_comparisons = []
    warnings = []

    available_groups = set(adata.obs[group_by].unique())
    available_splits = set(adata.obs[split_by].unique()) if split_by and split_by in adata.obs.columns else None

    for comp in comparisons:
        if len(comp) == 3:
            # Group comparison: ('group', 'A', 'B')
            comp_type, cat1, cat2 = comp
            if comp_type == 'group':
                if cat1 in available_groups and cat2 in available_groups:
                    valid_comparisons.append(comp)
                else:
                    missing = [c for c in [cat1, cat2] if c not in available_groups]
                    warnings.append(f"⚠️  Skipping group comparison {cat1} vs {cat2}: missing groups {missing}")
            else:
                warnings.append(f"⚠️  Unknown comparison type: {comp_type}")

        elif len(comp) == 4:
            # Split comparison: ('split', 'GroupX', 'Split1', 'Split2')
            comp_type, target_group, cat1, cat2 = comp
            if comp_type == 'split':
                # Check if target group exists
                if target_group not in available_groups:
                    warnings.append(f"⚠️  Skipping split comparison in {target_group}: group not found")
                    continue

                # Check if both splits exist within that group
                group_mask = adata.obs[group_by] == target_group
                group_data = adata.obs[group_mask]
                if len(group_data) == 0:
                    warnings.append(f"⚠️  Skipping split comparison in {target_group}: no cells found")
                    continue

                group_splits = set(group_data[split_by].unique()) if split_by in group_data.columns else set()

                if cat1 in group_splits and cat2 in group_splits:
                    # Check if both split categories have actual data
                    cat1_mask = (adata.obs[group_by] == target_group) & (adata.obs[split_by] == cat1)
                    cat2_mask = (adata.obs[group_by] == target_group) & (adata.obs[split_by] == cat2)

                    cat1_count = cat1_mask.sum()
                    cat2_count = cat2_mask.sum()

                    if cat1_count > 0 and cat2_count > 0:
                        valid_comparisons.append(comp)
                    else:
                        empty_cats = []
                        if cat1_count == 0:
                            empty_cats.append(cat1)
                        if cat2_count == 0:
                            empty_cats.append(cat2)
                        warnings.append(f"⚠️  Skipping split comparison {cat1} vs {cat2} in {target_group}: no cells in {empty_cats}")
                else:
                    missing = [c for c in [cat1, cat2] if c not in group_splits]
                    warnings.append(f"⚠️  Skipping split comparison {cat1} vs {cat2} in {target_group}: missing splits {missing}")
            else:
                warnings.append(f"⚠️  Unknown comparison type: {comp_type}")
        else:
            warnings.append(f"⚠️  Invalid comparison format: {comp}")

    return valid_comparisons, warnings


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


def _plot_single_facet(ax, ax2, facet_data, groups, group_colors_list,
                      split_by, split_colors, splits, jitter_points, jitter_dot_size, plot_mean,
                      mean_color, mean_size, fraction_df, fraction_threshold,
                      show_fraction, number_fontsize, number_decimal_places,
                      expression_threshold, show_xlabel, xlabel_rotation,
                      xlabel_ha, xlabel_fontsize, ylabel_fontsize, ylabel_mean_fontsize,
                      axis_tick_fontsize, plot_mean_pos_frac, mean_pos_frac_color, mean_pos_frac_size,
                      group_labels,
                      gene, layer, is_leftmost_subplot, is_rightmost_subplot,
                      comparison_stats=None, detailed_stats=False, proba_de_thresholds=(0.6, 0.8, 0.95), de_mode='change'):
    """Helper function to plot a single facet or the main plot"""

    # Create violin plots (with or without splits)
    if split_by is None or splits is None:
        # Original single violin logic
        for i, group in enumerate(groups):
            group_data = facet_data[facet_data['group'] == group]['expression']

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

                # Add jitter points
                if jitter_points:
                    x_jitter = np.random.normal(i, 0.1, size=len(group_data))
                    if isinstance(group_color, str):
                        group_color_rgb = mcolors.to_rgb(group_color)
                    else:
                        group_color_rgb = group_color[:3]
                    darker_color = tuple(c * 0.7 for c in group_color_rgb)
                    ax.scatter(x_jitter, group_data,
                              c=[darker_color], s=jitter_dot_size, alpha=0.8, edgecolors='none')

                # Add mean expression points if enabled
                if plot_mean:
                    group_mean = group_data.mean()
                    if ax2 is not None:
                        ax2.scatter(i, group_mean, c=mean_color, s=mean_size, marker='o',
                                   edgecolors='white', linewidth=1, zorder=10)

                    # Add expressing-cells-only mean if enabled
                    if plot_mean_pos_frac:
                        if fraction_df is not None:
                            # Use raw data for identifying expressing cells
                            group_indices = facet_data[facet_data['group'] == group].index
                            # Subset to indices that exist in the current facet
                            available_indices = fraction_df.index.intersection(group_indices)
                            if len(available_indices) > 0:
                                group_fraction_data = fraction_df.loc[available_indices, 'fraction_expr']
                                expressing_mask = group_fraction_data > fraction_threshold
                                if expressing_mask.sum() > 0:
                                    # Get the corresponding expression data for expressing cells
                                    expressing_indices = available_indices[expressing_mask]
                                    facet_expressing_data = facet_data[facet_data.index.isin(expressing_indices)]['expression']
                                    group_mean_pos_frac = facet_expressing_data.mean()
                                else:
                                    group_mean_pos_frac = 0
                            else:
                                group_mean_pos_frac = 0
                        else:
                            # Fallback: use expression data itself
                            expressing_mask = group_data > expression_threshold
                            if expressing_mask.sum() > 0:
                                group_mean_pos_frac = group_data[expressing_mask].mean()
                            else:
                                group_mean_pos_frac = 0

                        if ax2 is not None and group_mean_pos_frac > 0:
                            ax2.scatter(i, group_mean_pos_frac, c=mean_pos_frac_color, s=mean_pos_frac_size, marker='o',
                                       edgecolors='white', linewidth=1, zorder=11)
            else:
                # No data available for this group - create invisible placeholder + text
                # 1. Create invisible violin to establish coordinates
                placeholder_violin = ax.violinplot([0], positions=[i], widths=0.6, showmeans=False, showextrema=False)
                # Make it completely invisible
                for pc in placeholder_violin['bodies']:
                    pc.set_alpha(0)

                # 2. Add "No data" text (now with correct y-limits)
                ax.text(i, ax.get_ylim()[1] * 0.5, 'No data', ha='center', va='center',
                       fontsize=14, color='gray', style='italic',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))

    else:
        # Split violin logic
        violin_width = 0.8 / len(splits)  # Divide space among splits

        for i, group in enumerate(groups):
            group_data_all = facet_data[facet_data['group'] == group]

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
                        # Calculate true mean of all cells for this split
                        split_mean = split_data.mean()

                        if ax2 is not None:
                            ax2.scatter(x_pos, split_mean, c=mean_color, s=mean_size, marker='o',
                                    edgecolors='white', linewidth=1, zorder=10)

                        # Add expressing-cells-only mean if enabled
                        if plot_mean_pos_frac:
                            if fraction_df is not None:
                                # Use raw data for identifying expressing cells
                                split_indices = group_data_all[group_data_all['split'] == split].index
                                # Subset to indices that exist in the current facet
                                available_indices = fraction_df.index.intersection(split_indices)
                                if len(available_indices) > 0:
                                    split_fraction_data = fraction_df.loc[available_indices, 'fraction_expr']
                                    expressing_mask = split_fraction_data > fraction_threshold
                                    if expressing_mask.sum() > 0:
                                        # Get the corresponding expression data for expressing cells
                                        expressing_indices = available_indices[expressing_mask]
                                        facet_split_expressing_data = facet_data[facet_data.index.isin(expressing_indices)]['expression']
                                        split_mean_pos_frac = facet_split_expressing_data.mean()
                                    else:
                                        split_mean_pos_frac = 0
                                else:
                                    split_mean_pos_frac = 0
                            else:
                                # Fallback: use expression data itself
                                expressing_mask = split_data > expression_threshold
                                if expressing_mask.sum() > 0:
                                    split_mean_pos_frac = split_data[expressing_mask].mean()
                                else:
                                    split_mean_pos_frac = 0

                            if ax2 is not None and split_mean_pos_frac > 0:
                                ax2.scatter(x_pos, split_mean_pos_frac, c=mean_pos_frac_color, s=mean_pos_frac_size, marker='o',
                                           edgecolors='white', linewidth=1, zorder=11)
                else:
                    # No data available for this group + split combination - create invisible placeholder + text
                    x_offset = (j - (len(splits) - 1) / 2) * violin_width
                    x_pos = i + x_offset

                    # 1. Create invisible violin to establish coordinates
                    placeholder_violin = ax.violinplot([0], positions=[x_pos], widths=violin_width * 0.8, showmeans=False, showextrema=False)
                    # Make it completely invisible
                    for pc in placeholder_violin['bodies']:
                        pc.set_alpha(0)

                    # 2. Add "No data" text (now with correct y-limits)
                    ax.text(x_pos, ax.get_ylim()[1] * 0.5, 'No data', ha='center', va='center',
                           fontsize=12, color='gray', style='italic',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.3))

    # Add fraction numbers if requested
    if show_fraction:
        if split_by is None or splits is None:
            # Regular fraction numbers for non-split violins
            for i, group in enumerate(groups):
                group_data = facet_data[facet_data['group'] == group]['expression']

                if len(group_data) > 0:
                    n_cells = len(group_data)
                    if fraction_df is not None:
                        # Use raw data for fraction calculation
                        group_indices = facet_data[facet_data['group'] == group].index
                        # Subset fraction_df to indices that exist in the current facet
                        available_indices = fraction_df.index.intersection(group_indices)
                        if len(available_indices) > 0:
                            group_fraction_data = fraction_df.loc[available_indices, 'fraction_expr']
                            n_expressing = (group_fraction_data > fraction_threshold).sum()
                            frac_expressing = n_expressing / n_cells if n_cells > 0 else 0
                        else:
                            # Fallback if no indices match
                            n_expressing = (group_data > expression_threshold).sum()
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
            # Split fraction numbers
            violin_width = 0.8 / len(splits)  # Same as violin positioning

            for i, group in enumerate(groups):
                group_data_all = facet_data[facet_data['group'] == group]

                for j, split in enumerate(splits):
                    split_data = group_data_all[group_data_all['split'] == split]['expression']
                    x_offset = (j - (len(splits) - 1) / 2) * violin_width
                    x_pos = i + x_offset

                    if len(split_data) > 0:
                        n_cells = len(split_data)
                        if fraction_df is not None:
                            # Use raw data for fraction calculation
                            split_indices = group_data_all[group_data_all['split'] == split].index
                            # Subset fraction_df to indices that exist in the current facet
                            available_indices = fraction_df.index.intersection(split_indices)
                            if len(available_indices) > 0:
                                split_fraction_data = fraction_df.loc[available_indices, 'fraction_expr']
                                n_expressing = (split_fraction_data > fraction_threshold).sum()
                                frac_expressing = n_expressing / n_cells if n_cells > 0 else 0
                            else:
                                # Fallback if no indices match
                                n_expressing = (split_data > expression_threshold).sum()
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
                        # Empty data indicators for split
                        ax.text(x_pos, -0.08, '0', ha='center', va='top',
                                fontsize=number_fontsize, weight='bold', color='gray',
                                transform=ax.get_xaxis_transform())
                        ax.text(x_pos, -0.12, '0', ha='center', va='top',
                                fontsize=number_fontsize-1, color='gray',
                                transform=ax.get_xaxis_transform())
                        ax.text(x_pos, -0.16, '--', ha='center', va='top',
                                fontsize=number_fontsize-1, color='gray',
                                transform=ax.get_xaxis_transform())

    # Set basic formatting
    ax.set_xticks(range(len(groups)))

    # Handle x-axis labels with dynamic positioning
    if show_xlabel:
        # Calculate label position based on whether fractions are shown
        if show_fraction:
            # Move labels further down to avoid overlap with fraction numbers
            label_y_position = -0.25  # Below the fraction numbers
        else:
            # Standard position
            label_y_position = -0.08

        # Determine x-axis labels (use abbreviations if provided)
        if group_labels is not None:
            # Use abbreviated labels when group_labels is provided
            x_labels = [group_labels.get(group, group) for group in groups]
        else:
            # Use full group names
            x_labels = groups

        # Apply x-axis labels with custom formatting
        for i, (group, label) in enumerate(zip(groups, x_labels)):
            ax.text(i, label_y_position, label,
                   ha=xlabel_ha, va='top',
                   fontsize=xlabel_fontsize,
                   rotation=xlabel_rotation,
                   transform=ax.get_xaxis_transform())

        # Hide the default tick labels
        ax.set_xticklabels([])
    else:
        # Hide x-axis labels completely
        ax.set_xticklabels([])

    # Set y-axis labels with custom formatting
    if is_leftmost_subplot:
        # Primary y-axis label (expression) - only on leftmost subplots
        y_label = f'{gene} Expression'
        if layer is not None:
            y_label += f' ({layer})'
        ax.set_ylabel(y_label, fontsize=ylabel_fontsize)

    # Apply color and fontsize to all right y-axes
    if ax2 is not None and plot_mean:
        ax2.tick_params(axis='y', colors=mean_color, labelsize=axis_tick_fontsize)

    if is_rightmost_subplot:
        # Secondary y-axis label (mean expression) - only on rightmost subplots
        if ax2 is not None and plot_mean:
            ax2.set_ylabel('Mean Expression', fontsize=ylabel_mean_fontsize, color=mean_color)

    # Apply axis tick font sizes to all subplots
    ax.tick_params(axis='y', labelsize=axis_tick_fontsize)
    ax.tick_params(axis='x', labelsize=xlabel_fontsize)

    # Add statistical annotations for faceted plots
    if comparison_stats:
        y_min, y_max = ax.get_ylim()

        # Calculate annotation heights
        annotation_height_start = y_max * 1.05
        annotation_height_step = y_max * 0.08
        comparison_annotation_count = 0

        for comp_key, stats in comparison_stats.items():
            proba_de = stats.get('proba_de', 0)
            lfc = stats.get('lfc_mean', 0)  # Use lfc_mean instead of lfc
            bayes_factor = stats.get('bayes_factor', 0)

            # Determine significance level
            significance = _get_star_annotation(proba_de, proba_de_thresholds)

            # Always annotate all comparisons (including ns)
            # Parse comparison key to get groups
            if comp_key.startswith('group_'):
                # Group comparison: extract group names
                parts = comp_key.replace('group_', '').split('_vs_')
                if len(parts) == 2:
                    group1, group2 = parts

                    try:
                        x1 = groups.index(group1)
                        x2 = groups.index(group2)
                        y_pos = annotation_height_start + comparison_annotation_count * annotation_height_step

                        # Add significance line
                        ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1)
                        ax.plot([x1, x1], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)
                        ax.plot([x2, x2], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)

                        # Create annotation text based on detailed_stats setting
                        x_center = (x1 + x2) / 2

                        if detailed_stats == "full":
                            annotation = f"{significance}\nP(DE)={proba_de:.2f}\nLFC={lfc:.2f}\nBF={bayes_factor:.1f}"
                        elif detailed_stats == "medium":
                            annotation = f"{significance}\nP(DE)={proba_de:.2f}\nLFC={lfc:.2f}"
                        elif detailed_stats == "minimal":
                            annotation = f"{significance}\nP(DE)={proba_de:.2f}"
                        elif detailed_stats is True:  # Legacy support
                            if abs(lfc) >= 0.1:
                                annotation = f"{significance}\nP(DE)={proba_de:.2f}\nLFC={lfc:.2f}"
                            else:
                                annotation = f"{significance}\nP(DE)={proba_de:.2f}"
                        else:
                            annotation = significance

                        ax.text(x_center, y_pos + y_max*0.03, annotation, ha='center', va='bottom',
                               fontsize=10, fontweight='bold')
                        comparison_annotation_count += 1
                    except ValueError:
                        # Groups not found in this facet - skip annotation
                        pass

            elif comp_key.startswith('split_'):
                    # Split comparison: extract group and split names
                    # Format: split_<group>_<split1>_vs_<split2>
                    parts = comp_key.replace('split_', '').split('_vs_')
                    if len(parts) == 2:
                        left_part, split2 = parts
                        # Extract group and split1 from left part
                        # Need to find where group ends and split1 begins
                        for split in splits:
                            if left_part.endswith(f'_{split}'):
                                group = left_part[:-len(f'_{split}')]
                                split1 = split
                                break
                        else:
                            continue  # Skip if can't parse

                        # Find positions of the splits within this group
                        try:
                            group_idx = groups.index(group)

                            # Calculate x positions for split violins within this group
                            if len(splits) > 1:
                                violin_width = 0.6
                                split_width = violin_width / len(splits)

                                # Find split indices
                                split1_idx = splits.index(split1)
                                split2_idx = splits.index(split2)

                                # Calculate x positions
                                x1 = group_idx - (violin_width/2) + (split1_idx + 0.5) * split_width
                                x2 = group_idx - (violin_width/2) + (split2_idx + 0.5) * split_width

                                y_pos = annotation_height_start + comparison_annotation_count * annotation_height_step

                                # Add significance line
                                ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1)
                                ax.plot([x1, x1], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)
                                ax.plot([x2, x2], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)

                                # Create annotation text
                                x_center = (x1 + x2) / 2

                                if detailed_stats == "full":
                                    annotation = f"{significance}\nP(DE)={proba_de:.2f}\nLFC={lfc:.2f}\nBF={bayes_factor:.1f}"
                                elif detailed_stats == "medium":
                                    annotation = f"{significance}\nP(DE)={proba_de:.2f}\nLFC={lfc:.2f}"
                                elif detailed_stats == "minimal":
                                    annotation = f"{significance}\nP(DE)={proba_de:.2f}"
                                elif detailed_stats is True:  # Legacy support
                                    if abs(lfc) >= 0.1:
                                        annotation = f"{significance}\nP(DE)={proba_de:.2f}\nLFC={lfc:.2f}"
                                    else:
                                        annotation = f"{significance}\nP(DE)={proba_de:.2f}"
                                else:
                                    annotation = significance

                                ax.text(x_center, y_pos + y_max*0.03, annotation, ha='center', va='bottom',
                                       fontsize=10, fontweight='bold')
                                comparison_annotation_count += 1

                        except ValueError:
                            # Group not found in this facet - skip annotation
                            pass

        # Adjust y-axis to accommodate annotations if any were added
        if comparison_annotation_count > 0:
            max_annotation_height = annotation_height_start + comparison_annotation_count * annotation_height_step
            new_y_max = max_annotation_height + y_max * 0.1
            ax.set_ylim(0, new_y_max)


def _plot_single_facet_standard(ax, ax2, facet_data, groups, group_colors_list,
                               split_by, split_colors_list, splits, jitter_points, jitter_dot_size,
                               plot_mean, mean_color, mean_size, show_fraction,
                               expression_threshold, number_fontsize, number_decimal_places,
                               show_xlabel, xlabel_rotation, xlabel_ha, xlabel_fontsize,
                               ylabel_fontsize, ylabel_mean_fontsize, axis_tick_fontsize,
                               plot_mean_pos_frac, mean_pos_frac_color, mean_pos_frac_size,
                               group_labels, gene, layer, is_leftmost_subplot, is_rightmost_subplot,
                               comparison_stats=None, detailed_stats=False, stat_method='ttest',
                               stat_display_options=None, p_thresholds=(0.05, 0.01, 0.001)):
    """
    Plot a single facet with standard statistical framework (no scVI dependencies).

    Parameters
    ----------
    ax, ax2 : matplotlib.axes.Axes
        Primary and secondary (mean) axes
    facet_data : pd.DataFrame
        Data subset for this facet with columns: 'group', 'expression', optionally 'split'
    groups : list
        Ordered list of groups to plot
    group_colors_list : list
        Colors for each group
    split_by : str or None
        Column name for split violins
    split_colors_list : list or None
        Colors for split categories
    splits : list or None
        Ordered list of split categories
    jitter_points : bool
        Whether to add jitter points
    jitter_dot_size : float
        Size of jitter points
    plot_mean : bool
        Whether to plot mean dots
    mean_color : str
        Color for mean dots
    mean_size : float
        Size of mean dots
    show_fraction : bool
        Whether to show % expressing fractions
    expression_threshold : float
        Threshold for calculating % expressing
    number_fontsize : float
        Font size for numbers
    number_decimal_places : int
        Decimal places for fractions
    show_xlabel : bool
        Whether to show x-axis labels
    xlabel_rotation : float
        X-axis label rotation
    xlabel_ha : str
        X-axis label horizontal alignment
    xlabel_fontsize : float
        X-axis label font size
    ylabel_fontsize : float
        Y-axis label font size
    ylabel_mean_fontsize : float
        Mean y-axis label font size
    axis_tick_fontsize : float
        Axis tick font size
    plot_mean_pos_frac : bool
        Whether to plot expressing-cells-only means
    mean_pos_frac_color : str
        Color for expressing-cells-only means
    mean_pos_frac_size : float
        Size for expressing-cells-only means
    group_labels : dict or None
        Custom labels for groups
    gene : str
        Gene name
    layer : str or None
        Layer name for title
    is_leftmost_subplot : bool
        Whether this is the leftmost subplot
    is_rightmost_subplot : bool
        Whether this is the rightmost subplot
    comparison_stats : list or None
        Statistical comparisons to annotate
    detailed_stats : bool
        Whether to show detailed statistical labels
    stat_method : str
        Statistical method ('ttest', 'wilcoxon', 'mannwhitney')
    stat_display_options : dict or None
        Options for statistical display
    p_thresholds : tuple
        P-value thresholds for significance
    """
    import numpy as np

    # Default statistical display options
    if stat_display_options is None:
        stat_display_options = {
            'show_pvalue': False,
            'show_logfc': False,
            'show_statistic': False,
            'show_stars_only': True,
            'show_ns': False
        }

    # Create violin plots (with or without splits)
    if split_by is None or splits is None:
        # Regular violin plots
        for i, group in enumerate(groups):
            group_data = facet_data[facet_data['group'] == group]['expression']

            if len(group_data) > 0:
                # Create violin plot
                violin_parts = ax.violinplot([group_data], positions=[i], widths=0.6,
                                           showmeans=False, showextrema=False)

                # Apply colors
                group_color = group_colors_list[i % len(group_colors_list)]
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(group_color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(0.5)

                # Add jitter points
                if jitter_points:
                    x_jitter = np.random.normal(i, 0.08, size=len(group_data))
                    # Make jitter points darker
                    if isinstance(group_color, str):
                        import matplotlib.colors as mcolors
                        group_color_rgb = mcolors.to_rgb(group_color)
                    else:
                        group_color_rgb = group_color[:3]
                    darker_color = tuple(c * 0.7 for c in group_color_rgb)
                    ax.scatter(x_jitter, group_data, c=[darker_color], s=jitter_dot_size,
                             alpha=0.8, edgecolors='none')

                # Add mean expression dot
                if plot_mean and ax2 is not None:
                    mean_expr = group_data.mean()
                    ax2.scatter(i, mean_expr, c=mean_color, s=mean_size, marker='o',
                              edgecolors='white', linewidth=1, zorder=10)

                    # Add expressing-cells-only mean
                    if plot_mean_pos_frac:
                        expressing_cells = group_data[group_data > expression_threshold]
                        if len(expressing_cells) > 0:
                            mean_pos_frac = expressing_cells.mean()
                            ax2.scatter(i, mean_pos_frac, c=mean_pos_frac_color, s=mean_pos_frac_size,
                                      marker='o', edgecolors='white', linewidth=1, zorder=11)

                # Add fraction label in vlnplot_scvi format
                if show_fraction:
                    expressing_cells = group_data[group_data > expression_threshold]
                    total_cells = len(group_data)
                    expressing_count = len(expressing_cells)
                    fraction = expressing_count / total_cells if total_cells > 0 else 0

                    # Total cells - top row
                    ax.text(i, -0.08, f'{total_cells}', ha='center', va='top',
                            fontsize=number_fontsize, weight='bold',
                            transform=ax.get_xaxis_transform())
                    # Expressing cells - middle row
                    ax.text(i, -0.12, f'{expressing_count}', ha='center', va='top',
                            fontsize=number_fontsize-1,
                            transform=ax.get_xaxis_transform())
                    # Fraction expressing as decimal - bottom row
                    frac_text = f'{fraction:.{number_decimal_places}f}'
                    ax.text(i, -0.16, frac_text, ha='center', va='top',
                            fontsize=number_fontsize-1,
                            transform=ax.get_xaxis_transform())

            else:
                # Draw empty placeholder
                _draw_empty_placeholder_clean(ax, i, text="No data")

    else:
        # Split violin plots
        for i, group in enumerate(groups):
            group_data = facet_data[facet_data['group'] == group]

            if len(group_data) == 0:
                _draw_empty_placeholder_clean(ax, i, text="No data")
                continue

            # Calculate positions for split violins
            n_splits = len(splits)
            violin_width = 0.6
            split_width = violin_width / n_splits
            start_pos = i - violin_width/2 + split_width/2

            for j, split in enumerate(splits):
                split_data = group_data[group_data['split'] == split]['expression']
                x_pos = start_pos + j * split_width

                # Define split color for both data and no-data cases
                split_color = split_colors_list[j % len(split_colors_list)] if split_colors_list else group_colors_list[i % len(group_colors_list)]

                if len(split_data) > 0:
                    # Create split violin
                    violin_parts = ax.violinplot([split_data], positions=[x_pos], widths=split_width,
                                               showmeans=False, showextrema=False)
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(split_color)
                        pc.set_alpha(0.7)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(0.5)

                    # Add jitter points for splits
                    if jitter_points:
                        x_jitter = np.random.normal(x_pos, split_width * 0.1, size=len(split_data))
                        if isinstance(split_color, str):
                            import matplotlib.colors as mcolors
                            split_color_rgb = mcolors.to_rgb(split_color)
                        else:
                            split_color_rgb = split_color[:3]
                        darker_color = tuple(c * 0.7 for c in split_color_rgb)
                        ax.scatter(x_jitter, split_data, c=[darker_color], s=jitter_dot_size,
                                 alpha=0.8, edgecolors='none')

                    # Add mean for splits
                    if plot_mean and ax2 is not None:
                        mean_expr = split_data.mean()
                        ax2.scatter(x_pos, mean_expr, c=mean_color, s=mean_size, marker='o',
                                  edgecolors='white', linewidth=1, zorder=10)

                        # Add expressing-cells-only mean for splits
                        if plot_mean_pos_frac:
                            expressing_cells = split_data[split_data > expression_threshold]
                            if len(expressing_cells) > 0:
                                mean_pos_frac = expressing_cells.mean()
                                ax2.scatter(x_pos, mean_pos_frac, c=mean_pos_frac_color, s=mean_pos_frac_size,
                                          marker='o', edgecolors='white', linewidth=1, zorder=11)

                    # Add fraction labels for splits in vlnplot_scvi format
                    if show_fraction:
                        expressing_cells = split_data[split_data > expression_threshold]
                        total_cells = len(split_data)
                        expressing_count = len(expressing_cells)
                        fraction = expressing_count / total_cells if total_cells > 0 else 0

                        # Total cells - top row
                        ax.text(x_pos, -0.08, f'{total_cells}', ha='center', va='top',
                                fontsize=number_fontsize, weight='bold',
                                transform=ax.get_xaxis_transform())
                        # Expressing cells - middle row
                        ax.text(x_pos, -0.12, f'{expressing_count}', ha='center', va='top',
                                fontsize=number_fontsize-1,
                                transform=ax.get_xaxis_transform())
                        # Fraction expressing as decimal - bottom row
                        frac_text = f'{fraction:.{number_decimal_places}f}'
                        ax.text(x_pos, -0.16, frac_text, ha='center', va='top',
                                fontsize=number_fontsize-1,
                                transform=ax.get_xaxis_transform())

                else:
                    # Empty split placeholder
                    _draw_empty_split_placeholder(ax, x_pos, split_width, split_color)

    # Add statistical annotations
    if comparison_stats and len(comparison_stats) > 0:
        y_max = max(facet_data['expression'].max(), ax.get_ylim()[1])
        y_step = (y_max - ax.get_ylim()[0]) * 0.1

        for idx, comparison in enumerate(comparison_stats):
            # Extract comparison information
            if len(comparison) >= 3:
                _, group1, group2 = comparison[:3]

                try:
                    x1 = groups.index(group1)
                    x2 = groups.index(group2)

                    # Get data for statistical test
                    group1_data = facet_data[facet_data['group'] == group1]['expression']
                    group2_data = facet_data[facet_data['group'] == group2]['expression']

                    if len(group1_data) > 0 and len(group2_data) > 0:
                        # Compute statistics
                        stats_dict = _compute_standard_statistics(group1_data, group2_data, method=stat_method)

                        # Add annotation
                        y_pos = y_max + (idx + 1) * y_step
                        _add_statistical_annotation_standard(
                            ax, x1, x2, y_pos, stats_dict, stat_display_options, detailed_stats, p_thresholds
                        )

                except (ValueError, IndexError):
                    continue  # Skip invalid comparisons

    # Set axis labels and formatting
    if show_xlabel:
        if group_labels:
            labels = [group_labels.get(group, group) for group in groups]
        else:
            labels = groups
        ax.set_xticks(range(len(groups)))
        # Adjust x-label position based on whether fractions are shown
        if show_fraction:
            # Move x-labels down slightly when fractions are displayed
            ax.set_xticklabels(labels, rotation=xlabel_rotation, ha=xlabel_ha, fontsize=xlabel_fontsize)
            # Use tick_params to adjust label position
            ax.tick_params(axis='x', pad=8)  # Moderate padding when fractions are shown
        else:
            ax.set_xticklabels(labels, rotation=xlabel_rotation, ha=xlabel_ha, fontsize=xlabel_fontsize)
            ax.tick_params(axis='x', pad=3)   # Default padding when no fractions
    else:
        ax.set_xticks([])

    # Set y-axis labels
    if is_leftmost_subplot:
        layer_text = f" ({layer})" if layer else ""
        ax.set_ylabel(f'{gene} Expression{layer_text}', fontsize=ylabel_fontsize)

    # Set y2-axis label on rightmost subplot
    if is_rightmost_subplot and ax2 is not None and plot_mean:
        ax2.set_ylabel('Mean Expression', fontsize=ylabel_mean_fontsize, color=mean_color)
        ax2.tick_params(axis='y', colors=mean_color)

    # Set tick font sizes and colors separately for x and y axes
    ax.tick_params(axis='y', which='major', labelsize=axis_tick_fontsize)  # Y-axis ticks
    ax.tick_params(axis='x', which='major', labelsize=xlabel_fontsize)     # X-axis labels keep their size
    if ax2 is not None:
        ax2.tick_params(axis='y', which='major', labelsize=axis_tick_fontsize)
        if plot_mean:
            # Set y2 axis color to match mean color
            ax2.tick_params(axis='y', colors=mean_color)
            ax2.yaxis.label.set_color(mean_color)

    # Hide right y-axis label (but keep ticks) if not rightmost subplot
    if ax2 is not None and not is_rightmost_subplot:
        ax2.set_ylabel('')


def vlnplot(adata, gene, group_by,
           title=None,
           layer=None,
           expression_threshold=0.0,
           split_by=None,
           facet_by=None,
           group_order=None,
           split_order=None,
           facet_order=None,
           group_colors=None,
           split_colors=None,
           jitter_points=True,
           jitter_dot_size=12,
           plot_mean=True,
           show_fraction=True,
           show_legend=None,             # Smart default: False for regular, True for split
           legend_loc='upper right',     # Legend position
           show_group_legend=None,       # Explicit group legend control
           group_legend_loc='below',     # Group legend position
           group_legend_fontsize=None,   # Inherits from legend_fontsize
           show_xlabel=True,             # Control x-axis label visibility
           group_labels=None,            # X-axis abbreviation dictionary
           stat_method='ttest',          # Statistical method: 'ttest', 'wilcoxon', 'mannwhitney'
           comparisons=None,             # Simple comparison tuples: [('group', 'A', 'B')]
           detailed_stats=False,         # Show detailed statistical labels vs clean stars
           p_thresholds=(0.05, 0.01, 0.001),  # P-value thresholds for *, **, ***
           stat_display_options=None,    # Statistical display preferences
           title_fontsize=14,            # Main plot title
           subtitle_fontsize=9,          # Individual subplot titles
           ylabel_fontsize=10,           # Y-axis labels (left side)
           ylabel_mean_fontsize=10,      # Mean expression y-axis labels (right side)
           xlabel_fontsize=8,            # X-axis group labels
           axis_tick_fontsize=8,         # Axis tick numbers
           legend_fontsize=8,            # Split legend
           number_fontsize=6,            # Cell count/fraction numbers
           number_decimal_places=2,
           xlabel_rotation=45,
           xlabel_ha='right',
           figsize=(12, 8),
           facet_ncols=None,
           mean_color='black',
           mean_size=60,                 # Size for regular mean expression dots
           plot_mean_pos_frac=False,     # Enable expressing-cells-only mean dots
           mean_pos_frac_color='red',    # Color for expressing-cells-only mean dots
           mean_pos_frac_size=60,        # Size for expressing-cells-only mean dots
           free_y=True,
           free_mean_y=False,
           ylim=None):
    """
    Create violin plots for single-cell RNA sequencing data with standard statistical analysis.

    This function creates publication-ready violin plots optimized for standard single-cell
    data types (raw counts, CPM, log-normalized) with comprehensive visualization features:

    ✅ FULLY IMPLEMENTED FEATURES:
    • Split violins for treatment comparisons (split_by)
    • Multi-panel faceting by categories (facet_by)
    • Flexible y-axis scaling (free_y, free_mean_y)
    • Manual y-axis limits (ylim)
    • Dual legends (group colors + split colors)
    • Comprehensive font size controls
    • Cell fraction display (total, expressing, %)
    • Mean expression visualization with dual y-axes
    • Jitter point overlay
    • Custom styling and positioning

    ⚠️  STATISTICAL ANNOTATIONS NOT YET IMPLEMENTED:
    Statistical testing is performed (t-test, Wilcoxon, Mann-Whitney) but significance
    annotations (*, **, ***) are not yet drawn on plots. For statistical comparisons
    with visual annotations, use vlnplot_scvi() which includes scVI-based differential
    expression analysis with full annotation support.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with expression data
    gene : str
        Gene name to plot (must be in adata.var_names)
    group_by : str
        Column in adata.obs to group cells by
    title : str, optional
        Plot title. If None, uses gene name
    layer : str, optional
        Layer to use for expression data. If None, uses adata.X
    expression_threshold : float, default 0.0
        Threshold for calculating % expressing cells
    split_by : str, optional
        Column in adata.obs for split violins (e.g., treatment conditions)
    facet_by : str, optional
        Column in adata.obs for faceting plots
    group_order : list, optional
        Custom order for groups
    split_order : list, optional
        Custom order for split categories
    facet_order : list, optional
        Custom order for facet categories
    group_colors : dict, optional
        Custom colors for groups {group_name: color}
    split_colors : dict or list, optional
        Custom colors for split categories
    jitter_points : bool, default True
        Whether to show individual data points
    jitter_dot_size : float, default 12
        Size of jitter points
    plot_mean : bool, default True
        Whether to show mean expression dots
    show_fraction : bool, default True
        Whether to show cell counts and % expressing
    show_legend : bool, optional
        Whether to show split legend. Auto-determined if None
    legend_loc : str, default 'upper right'
        Legend position
    show_group_legend : bool, optional
        Whether to show group legend
    group_legend_loc : str, default 'below'
        Group legend position
    group_legend_fontsize : float, optional
        Group legend font size
    show_xlabel : bool, default True
        Whether to show x-axis labels
    group_labels : dict, optional
        Custom x-axis labels {group_name: label}
    stat_method : str, default 'ttest'
        Statistical method: 'ttest', 'wilcoxon', 'mannwhitney'
    comparisons : list, optional
        Statistical comparisons: [('group', 'A', 'B'), ...]
    detailed_stats : bool, default False
        Show detailed stats (t=2.45, p=0.023) vs stars (**)
    p_thresholds : tuple, default (0.05, 0.01, 0.001)
        P-value thresholds for *, **, ***
    stat_display_options : dict, optional
        Statistical display preferences:
        - show_pvalue: Show p-values
        - show_logfc: Show log fold changes
        - show_statistic: Show test statistics
        - show_stars_only: Show only significance stars
        - show_ns: Show 'ns' for non-significant
    title_fontsize : float, default 14
        Title font size
    subtitle_fontsize : float, default 9
        Subplot title font size
    ylabel_fontsize : float, default 10
        Y-axis label font size
    ylabel_mean_fontsize : float, default 10
        Mean y-axis label font size
    xlabel_fontsize : float, default 8
        X-axis label font size
    axis_tick_fontsize : float, default 8
        Axis tick font size
    legend_fontsize : float, default 8
        Legend font size
    number_fontsize : float, default 6
        Cell count/fraction font size
    number_decimal_places : int, default 2
        Decimal places for fractions
    xlabel_rotation : float, default 45
        X-axis label rotation
    xlabel_ha : str, default 'right'
        X-axis label horizontal alignment
    figsize : tuple, default (12, 8)
        Figure size
    facet_ncols : int, optional
        Number of columns for faceted plots
    mean_color : str, default 'black'
        Color for mean dots
    mean_size : float, default 60
        Size for mean dots
    plot_mean_pos_frac : bool, default False
        Show expressing-cells-only means
    mean_pos_frac_color : str, default 'red'
        Color for expressing-cells-only means
    mean_pos_frac_size : float, default 60
        Size for expressing-cells-only means
    free_y : bool, default True
        Independent y-axes for facets
    free_mean_y : bool, default False
        Independent mean y-axes for facets
    ylim : tuple, optional
        Y-axis limits

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object

    Examples
    --------
    Basic violin plot with group legend:
    >>> fig = vlnplot(adata, 'GNLY', 'condition',
    ...               group_colors={'Control': 'blue', 'Treatment': 'red'},
    ...               show_group_legend=True, group_legend_loc='below')

    Split violin plot for treatment comparisons:
    >>> fig = vlnplot(adata, 'GNLY', 'condition', split_by='treatment',
    ...               split_colors={'pre': 'red', 'post': 'blue'},
    ...               show_legend=True, legend_loc='upper right')

    Multi-panel faceted plot with custom styling:
    >>> fig = vlnplot(adata, 'GNLY', 'condition', facet_by='subject',
    ...               facet_ncols=2, free_y=True,
    ...               xlabel_fontsize=12, ylabel_fontsize=14,
    ...               show_fraction=True, plot_mean_pos_frac=True)

    Combined split + facet plot (like your current example):
    >>> fig = vlnplot(adata, 'GNLY', 'condition',
    ...               split_by='treatment', facet_by='subject',
    ...               group_labels={'Nonlesional':'NL', 'SADBE':'S'},
    ...               ylim=(0, 1000), figsize=(14,14))

    Raw count data with appropriate threshold:
    >>> fig = vlnplot(adata, 'GNLY', 'condition', layer='raw',
    ...               expression_threshold=0.0,  # Raw counts
    ...               jitter_points=True, mean_color='blue')
    """
    import math
    import pandas as pd
    import numpy as np
    import seaborn as sns

    # Input validation
    if adata is None:
        raise ValueError("AnnData object cannot be None")
    if gene is None:
        raise ValueError("Gene cannot be None")
    if group_by is None:
        raise ValueError("group_by cannot be None")

    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")
    if group_by not in adata.obs.columns:
        raise ValueError(f"Group column '{group_by}' not found in adata.obs")

    # Extract expression data
    if layer is None:
        print("⚠️  Using .X matrix - consider specifying a layer for clearer data interpretation")
        expression_data = adata[:, gene].X
        data_source = "adata.X"
    else:
        if layer not in adata.layers.keys():
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        expression_data = adata[:, gene].layers[layer]
        data_source = f"layer '{layer}'"

    # Convert to dense if sparse
    if hasattr(expression_data, 'toarray'):
        expression_data = expression_data.toarray()
    expression_data = expression_data.flatten()

    print(f"📊 Using data from: {data_source}")
    print(f"📈 Expression range: {expression_data.min():.3f} - {expression_data.max():.3f}")

    # Build main DataFrame
    plot_data = pd.DataFrame({
        'group': adata.obs[group_by].values,
        'expression': expression_data
    }, index=adata.obs.index)

    # Add split and facet columns if specified
    if split_by is not None:
        if split_by not in adata.obs.columns:
            raise ValueError(f"Split column '{split_by}' not found in adata.obs")
        plot_data['split'] = adata.obs[split_by].values

    if facet_by is not None:
        if facet_by not in adata.obs.columns:
            raise ValueError(f"Facet column '{facet_by}' not found in adata.obs")
        plot_data['facet'] = adata.obs[facet_by].values

    # Validate and process comparisons
    if comparisons is not None:
        valid_comparisons, warnings = _validate_and_filter_comparisons(
            comparisons, adata, group_by, split_by
        )
        for warning in warnings:
            print(f"⚠️  {warning}")
        comparisons = valid_comparisons

    # Category ordering
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
                print(f"⚠️  {param_name}: Categories {missing_in_custom} not in custom order")

            valid_order = [cat for cat in custom_order if cat in available_categories]
            print(f"📋 {param_name} order: {valid_order}")
            return valid_order
        else:
            default_order = sorted(available_categories)
            print(f"📋 {param_name} order (alphabetical): {default_order}")
            return default_order

    # Apply category ordering
    groups = _get_category_order(plot_data['group'], group_order, 'Groups')

    if split_by is not None:
        splits = _get_category_order(plot_data['split'], split_order, 'Splits')
    else:
        splits = None

    # Set up colors
    group_colors_list = _get_colors_from_dict(
        group_colors, groups, sns.color_palette("Set2", len(groups)), "Group colors"
    )

    if split_by is not None and splits is not None:
        split_colors_list = _get_colors_from_dict(
            split_colors, splits, sns.color_palette("Set1", len(splits)), "Split colors"
        )
    else:
        split_colors_list = None

    # Layout determination
    if facet_by is not None:
        facets = _get_category_order(plot_data['facet'], facet_order, 'Facets')
        n_facets = len(facets)

        if facet_ncols is not None:
            facet_cols = min(facet_ncols, n_facets)
            facet_rows = math.ceil(n_facets / facet_cols)
        else:
            facet_cols = n_facets
            facet_rows = 1

        if figsize is None:
            figsize = (4 * facet_cols, 4 * facet_rows)

        fig, ax_pairs = _create_faceted_plot_layout(facet_rows, facet_cols, figsize)
        is_faceted = True
    else:
        facets = [None]
        fig, ax_pairs = _create_single_plot_layout(figsize)
        is_faceted = False

    # Default statistical display options
    if stat_display_options is None:
        stat_display_options = {
            'show_pvalue': False,
            'show_logfc': False,
            'show_statistic': False,
            'show_stars_only': True,
            'show_ns': False
        }

    # Plot each facet
    for idx, (ax1, ax2, row, col) in enumerate(ax_pairs):
        if is_faceted:
            if idx >= len(facets):
                ax1.axis('off')
                ax2.axis('off')
                continue

            facet_cat = facets[idx]
            facet_data = plot_data[plot_data['facet'] == facet_cat]
            subplot_title = f"{facet_cat}"
        else:
            facet_data = plot_data
            subplot_title = title if title else f"{gene} Expression"

        # Determine subplot positions for axis labeling
        is_leftmost = (col == 0) if is_faceted else True
        if is_faceted:
            # For faceted plots, rightmost is the last column OR the last subplot in the grid
            is_rightmost = (col == facet_cols - 1) or (idx == len(facets) - 1)
        else:
            is_rightmost = True

        # Call plotting function
        _plot_single_facet_standard(
            ax1, ax2, facet_data, groups, group_colors_list,
            split_by, split_colors_list, splits, jitter_points, jitter_dot_size,
            plot_mean, mean_color, mean_size, show_fraction,
            expression_threshold, number_fontsize, number_decimal_places,
            show_xlabel, xlabel_rotation, xlabel_ha, xlabel_fontsize,
            ylabel_fontsize, ylabel_mean_fontsize, axis_tick_fontsize,
            plot_mean_pos_frac, mean_pos_frac_color, mean_pos_frac_size,
            group_labels, gene, layer, is_leftmost, is_rightmost,
            comparisons, detailed_stats, stat_method,
            stat_display_options, p_thresholds
        )

        # Add subplot title
        if is_faceted:
            ax1.set_title(subplot_title, fontsize=subtitle_fontsize)

    # Set main title
    if not is_faceted and title:
        fig.suptitle(title, fontsize=title_fontsize)
    elif is_faceted:
        main_title = title if title else f"{gene} Expression"
        fig.suptitle(main_title, fontsize=title_fontsize)

    # Add legends if needed
    # Set default group legend fontsize
    if group_legend_fontsize is None:
        group_legend_fontsize = legend_fontsize

    # Determine which legends to show
    show_split_legend = show_legend and split_by is not None
    show_group_legend_flag = show_group_legend or (show_legend and split_by is None)

    # Create split legend if requested
    if show_split_legend:
        # Create legend elements for splits
        legend_elements = []
        for i, split in enumerate(splits):
            color = split_colors_list[i % len(split_colors_list)]
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color, markersize=8, label=split))

        fig.legend(handles=legend_elements, loc=legend_loc, fontsize=legend_fontsize,
                  title=split_by, title_fontsize=legend_fontsize)

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
            # Place below the entire figure
            group_legend = fig.legend(handles=group_legend_elements, fontsize=group_legend_fontsize,
                                   title=group_by, title_fontsize=group_legend_fontsize,
                                   bbox_to_anchor=(0.5, 0.02), loc='lower center',
                                   ncol=len(groups))
        elif group_legend_loc == 'right':
            # Place to the right of the entire figure
            group_legend = fig.legend(handles=group_legend_elements, fontsize=group_legend_fontsize,
                                   title=group_by, title_fontsize=group_legend_fontsize,
                                   bbox_to_anchor=(0.98, 0.5), loc='center left')
        else:
            # Use the first axes for positioning
            first_ax = ax_pairs[0][0] if ax_pairs else None
            if first_ax:
                group_legend = first_ax.legend(handles=group_legend_elements, loc=group_legend_loc,
                                             fontsize=group_legend_fontsize,
                                             title=group_by, title_fontsize=group_legend_fontsize)

    # Apply y-axis scaling for faceted plots
    if is_faceted:
        # Handle free_y scaling for expression data (left y-axis)
        if not free_y:
            # Calculate global y-limits across all facets
            all_y_mins, all_y_maxs = [], []
            for ax1, ax2, row, col in ax_pairs:
                y_min, y_max = ax1.get_ylim()
                all_y_mins.append(y_min)
                all_y_maxs.append(y_max)
            if all_y_mins and all_y_maxs:
                global_y_min = min(all_y_mins)
                global_y_max = max(all_y_maxs)
                # Apply global limits to all facets
                for ax1, ax2, row, col in ax_pairs:
                    ax1.set_ylim(global_y_min, global_y_max)

        # Handle free_mean_y scaling for mean expression data (right y-axis)
        if not free_mean_y and plot_mean:
            # Calculate global mean y-limits across all facets
            all_mean_y_mins, all_mean_y_maxs = [], []
            for ax1, ax2, row, col in ax_pairs:
                if ax2 is not None:
                    mean_y_min, mean_y_max = ax2.get_ylim()
                    all_mean_y_mins.append(mean_y_min)
                    all_mean_y_maxs.append(mean_y_max)
            if all_mean_y_mins and all_mean_y_maxs:
                global_mean_y_min = min(all_mean_y_mins)
                global_mean_y_max = max(all_mean_y_maxs)
                # Apply global limits to all mean axes
                for ax1, ax2, row, col in ax_pairs:
                    if ax2 is not None:
                        ax2.set_ylim(global_mean_y_min, global_mean_y_max)

    # Apply ylim override if specified (only affects expression/left y-axis)
    if ylim is not None:
        for ax1, ax2, row, col in ax_pairs:
            ax1.set_ylim(ylim)

    plt.tight_layout()

    return fig


def vlnplot_scvi(adata, gene, group_by,
                    title=None,
                    layer=None,
                    raw_layer='raw',
                    expression_threshold=0.1,
                    split_by=None,
                    facet_by=None,
                    group_order=None,
                    split_order=None,
                    facet_order=None,
                    group_colors=None,
                    split_colors=None,
                    jitter_points=True,
                    jitter_dot_size=12,
                    plot_mean=True,
                    show_fraction=True,
                    show_legend=None,            # Smart default: False for regular, True for split
                    legend_loc='upper right',    # Legend position
                    show_group_legend=None,      # Explicit group legend control
                    group_legend_loc='below',    # Group legend position
                    group_legend_fontsize=None,  # Inherits from legend_fontsize
                    show_xlabel=True,            # Control x-axis label visibility
                    group_labels=None,           # X-axis abbreviation dictionary
                    scvi_model=None,             # scVI model for differential expression
                    comparisons=None,            # Simple comparison tuples: [('group', 'A', 'B')]
                    detailed_stats=False,        # Show detailed P(DE) labels vs clean stars
                    proba_de_thresholds=(0.6, 0.8, 0.95),  # Thresholds for *, **, ***
                    de_mode='change',            # scVI DE mode: 'change' or 'vanilla'
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
                    facet_ncols=None,
                    mean_color='black',
                    mean_size=60,                     # Size for regular mean expression dots
                    plot_mean_pos_frac=False,        # Enable expressing-cells-only mean dots
                    mean_pos_frac_color='red',        # Color for expressing-cells-only mean dots
                    mean_pos_frac_size=60,            # Size for expressing-cells-only mean dots
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
    title : str, optional
        Custom plot title. If None (default), uses gene name. For faceted plots,
        becomes the main title above all subplots.
        Example: 'GNLY Expression Across Conditions'
    layer : str, optional
        scVI-transformed expression data layer to use instead of .X.
        Useful when scVI-transformed counts from different preprocessing
        (e.g. ambient RNA-corrected vs uncorrected) are stored in separate layers.
        Must exist in adata.layers. For raw/standard normalized counts,
        use the vlnplot() function instead.
        Example: 'scvi_corrected', 'scvi_uncorrected'
    split_by : str, optional
        Column name in adata.obs for split violin plots
        (e.g., 'treatment', 'genotype', 'stimulus', 'batch')
    facet_by : str, optional
        Column name in adata.obs for creating subplot facets (R's facet_wrap equivalent).
        Creates a separate subplot for each unique value in this column, arranged in a grid.
        When combined with comparisons and scvi_model, performs per-facet differential
        expression analysis for more granular statistical insights.
        Example: 'subject', 'timepoint', 'batch', 'experiment'
    raw_layer : str, default 'raw'
        Layer containing raw counts for fraction expressing calculation.
        Falls back to main data with threshold if layer not found.
    expression_threshold : float, default 0.1
        Threshold for determining expressing cells when using scVI data directly

    Ordering and Layout Parameters
    ------------------------------
    group_order : list, optional
        Custom order for group_by categories. If None, uses alphabetical order.
        Example: ['Control', 'Treatment', 'Recovery']
    split_order : list, optional
        Custom order for split_by categories. If None, uses alphabetical order.
        Example: ['pre', 'dupi', 'post']
    facet_order : list, optional
        Custom order for facet_by categories. If None, uses alphabetical order.
        Example: ['BB07', 'BB10', 'BB11', 'BB12']
    facet_ncols : int, optional
        Number of columns for facet grid layout. If None, uses horizontal layout.
        Arranges facets in n_cols × n_rows grid. Example: 4 subjects with
        facet_ncols=2 creates 2×2 grid.

    Statistical Analysis Parameters
    ------------------------------
    scvi_model : scvi model, optional
        Trained scVI model for differential expression analysis
    comparisons : list of tuples, optional
        Statistical comparisons to perform using scVI differential expression.
        Supports two comparison types:

        **Group comparisons (3-tuple)**: ('group', group1, group2)
        Compare between categories of group_by variable.

        **Split comparisons (4-tuple)**: ('split', within_group, split1, split2)
        Compare between categories of split_by variable within a specific group_by category.

        **Order matters**: first category is the reference, second is the comparison.

        Examples:
            # Group comparisons
            [('group', 'Control', 'Treatment')]           # Treatment vs Control
            [('group', 'Nonlesional', 'SADBE'),          # SADBE vs Nonlesional
             ('group', 'Nonlesional', 'Metal')]          # Metal vs Nonlesional

            # Split comparisons (requires split_by parameter)
            [('split', 'dupi', 'pre', 'post')]           # post vs pre within dupi group
            [('split', 'Nonlesional', 'Treatment', 'Control')] # Treatment vs Control within Nonlesional

        **When to use which**:
        - Use 'group' when comparing different categories of your main grouping variable
        - Use 'split' when comparing split_by categories within specific groups
        - Can mix both types in the same comparisons list

        **Statistics returned**:
        - proba_de: Probability of differential expression (0-1)
        - LFC (log fold change): log2(comparison/reference)
          * Positive LFC = comparison group higher than reference
          * Negative LFC = comparison group lower than reference
        - Bayes factor: Evidence strength for differential expression

        Requires scvi_model parameter.
    detailed_stats : bool or str, default False
        Control level of statistical detail in annotations:
        - False: Show only significance stars (*, **, ***)
        - True: Show stars + P(DE) + LFC (if >= 0.1) [backward compatible]
        - "minimal": Show stars + P(DE) only
        - "medium": Show stars + P(DE) + LFC (always)
        - "full": Show stars + P(DE) + LFC + Bayes factor

        Note: P(DE) is the Bayesian probability of differential expression (not a p-value)
    proba_de_thresholds : tuple, default (0.6, 0.8, 0.95)
        Probability thresholds for *, **, *** significance levels
    de_mode : str, default 'change'
        scVI differential expression mode: 'change' or 'vanilla'.
        'change' mode focuses on meaningful effect sizes and is generally recommended.
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
        Smart automatic legend display:
        - When split_by is used: shows split legend (colors for split categories)
        - When split_by is None: shows group legend (colors for x-axis groups)
        - None (default): auto-determines based on split_by presence
        - True/False: explicitly enable/disable automatic legend behavior
    show_group_legend : bool, optional
        Explicit control for group legend display (x-axis group colors):
        - None (default): follows show_legend automatic behavior
        - True: force display group legend regardless of split_by
        - False: never show group legend
        Works independently of show_legend for split plots.
    group_legend_loc : str, default 'below'
        Group legend position. Same options as legend_loc.
    group_legend_fontsize : int, optional
        Group legend font size. Inherits from legend_fontsize if not specified.
    legend_loc : str, default 'upper right'
        Legend position. Standard matplotlib locations: 'upper right', 'upper left',
        'lower left', 'lower right', 'right', 'center left', 'center right',
        'lower center', 'upper center', 'center', 'best'. Custom options: 'below'
        (places legend below plot), 'right' (places legend to right with spacing).
    xlabel_rotation : float, default 45
        Rotation angle for x-axis labels
    mean_size : int, default 60
        Size of regular mean expression dots
    plot_mean_pos_frac : bool, default False
        Enable expressing-cells-only mean dots (in addition to regular mean dots)
    mean_pos_frac_color : str, default 'red'
        Color for expressing-cells-only mean dots
    mean_pos_frac_size : int, default 60
        Size of expressing-cells-only mean dots
    free_y : bool, default True
        Allow independent y-axis scaling for each facet subplot.
        When True, each facet optimizes its y-axis range based on local data.
        When False, all facets share the same y-axis range for easier comparison.
    free_mean_y : bool, default False
        Allow independent mean expression y-axis (right side) scaling for each facet.
        When True, each facet optimizes its mean y-axis independently.
        When False, all facets share the same mean y-axis range.
    ylim : tuple, optional
        Fixed y-axis limits for expression data (left y-axis).
        Format: (y_min, y_max). Overrides free_y behavior when specified.
        Example: (0, 10) to set expression range from 0 to 10.
    *_fontsize : int
        Font sizes for various plot elements (title, labels, legend, etc.)

    Returns
    -------
    matplotlib.figure.Figure
        The generated violin plot figure

    Examples
    --------
    Pattern A: Split effects within groups (e.g., treatment effects within conditions)

    >>> # Statistical comparisons using the modern comparisons interface
    >>> ssc.vlnplot_scvi(
    ...     adata, 'GNLY', group_by='condition', split_by='treatment',
    ...     scvi_model=scvi_model,
    ...     comparisons=[
    ...         ('group', 'Nonlesional', 'SADBE'),
    ...         ('group', 'SADBE', 'Metal'),
    ...         ('split', 'Nonlesional', 'dupi', 'pre'),
    ...         ('split', 'SADBE', 'dupi', 'pre')
    ...     ],
    ...     detailed_stats='full',
    ...     proba_de_thresholds=(0.4, 0.6, 0.8)
    ... )

    Pattern B: Faceted plots with per-facet differential expression

    >>> # Faceted plot across subjects with per-facet statistical analysis
    >>> ssc.vlnplot_scvi(
    ...     adata, 'GNLY', group_by='condition', split_by='treatment',
    ...     facet_by='subject', facet_ncols=2,
    ...     scvi_model=scvi_model,
    ...     comparisons=[
    ...         ('group', 'Nonlesional', 'SADBE'),     # Group comparison per facet
    ...         ('split', 'Nonlesional', 'dupi', 'pre'), # Split comparison per facet
    ...         ('split', 'SADBE', 'dupi', 'pre')
    ...     ],
    ...     detailed_stats='medium',
    ...     facet_order=['BB07', 'BB10', 'BB11', 'BB12'],
    ...     figsize=(16, 12)
    ... )

    Notes
    -----
    - Statistical comparisons require scvi_model parameter
    - Use detailed_stats='full' to see * annotation, P(DE), log fold changes, and Bayes Factor on plot
    - Function automatically handles scVI's 1-vs-all DE results and converts to pairwise comparisons
    - Supports both group comparisons and split comparisons within groups
    - **Faceting behavior**: When facet_by is used with comparisons, differential expression
      analysis is performed separately for each facet (e.g., per-subject analysis).
      This enables detection of subject-specific or context-specific expression differences
      rather than population-level averages. Each subplot shows its own statistical annotations.
    """

    import warnings

    # Input validation
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in adata.var_names")
    if group_by not in adata.obs.columns:
        raise ValueError(f"Group column '{group_by}' not found in adata.obs")

    # Statistical framework setup
    if comparisons and not scvi_model:
        raise ValueError("scvi_model is required when comparisons are specified")


    # Process comparisons for statistical annotations
    if comparisons and scvi_model:
        # Validate comparisons before processing to avoid DE errors
        valid_comparisons, comp_warnings = _validate_and_filter_comparisons(comparisons, adata, group_by, split_by)

        # Print warnings about skipped comparisons
        for warning in comp_warnings:
            print(warning)

        if len(valid_comparisons) > 0:
            print(f"ℹ️  Processing {len(valid_comparisons)} valid comparisons out of {len(comparisons)} requested")
            comparison_stats = _process_comparisons(adata, gene, scvi_model, valid_comparisons, group_by, split_by)
        else:
            print("ℹ️  No valid comparisons found - proceeding without statistical annotations")
            comparison_stats = {}
    else:
        comparison_stats = {}

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

    # Storage for axis scaling
    all_means_pos_frac = []  # For expressing-cells-only means

    # Check if faceting is requested
    if facet_by is not None:
        # FACETED PLOT PATH
        print(f"📊 Creating faceted plot by: {facet_by}")

        # Add facet column to plot_data
        if facet_by not in adata.obs.columns:
            raise ValueError(f"Facet column '{facet_by}' not found in adata.obs")
        plot_data['facet_by'] = adata.obs[facet_by].values

        # Get facet categories in order
        if facet_order is not None:
            # Use custom order with validation
            facet_categories = [cat for cat in facet_order if cat in plot_data['facet_by'].unique()]
            # Add any missing categories not in custom order
            missing_categories = [cat for cat in plot_data['facet_by'].unique() if cat not in facet_order]
            facet_categories.extend(sorted(missing_categories))
            if missing_categories:
                print(f"⚠️  Facet order: Categories {missing_categories} not in facet_order, added at end")
        else:
            # Default alphabetical order
            facet_categories = sorted(plot_data['facet_by'].unique())
        n_facets = len(facet_categories)
        print(f"📋 Facet categories: {facet_categories}")

        # Calculate grid dimensions
        if facet_ncols is not None:
            n_cols = min(facet_ncols, n_facets)
            n_rows = (n_facets + n_cols - 1) // n_cols  # Ceiling division
        else:
            n_cols = n_facets  # Default: horizontal layout
            n_rows = 1

        # Create faceted figure - use figsize or auto-calculate
        if figsize is not None:
            final_figsize = figsize
        else:
            final_figsize = (4 * n_cols, 4 * n_rows)  # Auto-calculate based on grid

        fig, axes = plt.subplots(n_rows, n_cols, figsize=final_figsize, squeeze=False)

        # Handle split colors for faceted plots
        if split_by is not None and splits is not None and len(splits) > 0:
            # Get split colors using the same logic as single plots
            split_colors_dict = _get_colors_from_dict(split_colors, splits,
                                                     sns.color_palette("Set3", len(splits)),
                                                     "Split colors") if isinstance(split_colors, dict) else split_colors
        else:
            split_colors_dict = split_colors

        # Store axes for y-axis scaling
        facet_axes_pairs = []

        # Plot each facet
        for idx, facet_cat in enumerate(facet_categories):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            ax2 = ax.twinx()

            # Store axes pair for later y-axis scaling
            facet_axes_pairs.append((ax, ax2))

            # Subset data for this facet
            facet_data = plot_data[plot_data['facet_by'] == facet_cat].copy()

            # Calculate per-facet DE statistics if comparisons are specified
            facet_comparison_stats = {}
            if comparisons and scvi_model and len(facet_data) > 0:
                # Subset adata to this facet for DE analysis
                facet_mask = adata.obs[facet_by] == facet_cat
                facet_adata = adata[facet_mask].copy()

                if len(facet_adata) > 0:
                    print(f"🔬 Computing DE for facet {facet_cat} ({len(facet_adata)} cells)")

                    # Validate and filter comparisons for this facet
                    valid_facet_comparisons, _ = _validate_and_filter_comparisons(
                        comparisons, facet_adata, group_by, split_by)

                    if len(valid_facet_comparisons) > 0:
                        facet_comparison_stats = _process_comparisons(
                            facet_adata, gene, scvi_model, valid_facet_comparisons, group_by, split_by)
                        print(f"   ✓ Found {len(facet_comparison_stats)} significant comparisons in {facet_cat}")
                    else:
                        print(f"   ○ No valid comparisons for facet {facet_cat}")
                else:
                    print(f"   ⚠️ No data for facet {facet_cat}")

            # Set subplot title
            ax.set_title(f"{facet_cat}", fontsize=subtitle_fontsize)

            # Determine subplot position for y-axis label placement
            is_leftmost_subplot = (col == 0)
            is_rightmost_subplot = (col == n_cols - 1)

            # Reuse the existing single plot logic for this subplot
            _plot_single_facet(ax, ax2, facet_data, groups, group_colors_list,
                              split_by, split_colors_dict, splits, jitter_points, jitter_dot_size, plot_mean,
                              mean_color, mean_size, fraction_df, fraction_threshold,
                              show_fraction, number_fontsize, number_decimal_places,
                              expression_threshold, show_xlabel, xlabel_rotation,
                              xlabel_ha, xlabel_fontsize, ylabel_fontsize, ylabel_mean_fontsize,
                              axis_tick_fontsize, plot_mean_pos_frac, mean_pos_frac_color, mean_pos_frac_size,
                              group_labels,
                              gene, layer, is_leftmost_subplot, is_rightmost_subplot,
                              facet_comparison_stats, detailed_stats, proba_de_thresholds, de_mode)

        # Hide empty subplots
        for idx in range(n_facets, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        # Determine which legends to show for faceted plots (same logic as single plots)
        show_split_legend = show_legend and split_by is not None

        # Add legend for split violins in faceted plots
        if show_split_legend and splits is not None:
            # Create legend elements for splits using the last subplot for positioning
            last_ax = axes.flat[-1] if len(axes.flat) > 0 else axes[0, 0]

            split_legend_elements = []
            for j, split in enumerate(splits):
                if split_colors_dict and split in split_colors_dict:
                    split_color = split_colors_dict[split]
                else:
                    # Default colors if no custom split colors provided
                    default_colors = ['purple', 'orange', 'green', 'red', 'blue']
                    split_color = default_colors[j % len(default_colors)]

                split_legend_elements.append(plt.Rectangle((0,0), 1, 1,
                                                         facecolor=split_color, alpha=0.7,
                                                         edgecolor='black', linewidth=0.5,
                                                         label=split))

            # Handle custom legend positions for faceted plots
            if legend_loc == 'below':
                # Place below the entire figure
                split_legend = fig.legend(handles=split_legend_elements, fontsize=legend_fontsize,
                                       title=split_by, title_fontsize=legend_fontsize,
                                       bbox_to_anchor=(0.5, 0.02), loc='lower center',
                                       ncol=len(splits))
            elif legend_loc == 'right':
                # Place to the right of the entire figure
                split_legend = fig.legend(handles=split_legend_elements, fontsize=legend_fontsize,
                                       title=split_by, title_fontsize=legend_fontsize,
                                       bbox_to_anchor=(0.98, 0.5), loc='center left')
            else:
                # Place in specified location on the last subplot
                split_legend = last_ax.legend(handles=split_legend_elements, loc=legend_loc,
                                           fontsize=legend_fontsize,
                                           title=split_by, title_fontsize=legend_fontsize)

        # Add group legend for faceted plots (same logic as single plots)
        show_group_legend_flag = show_group_legend or (show_legend and split_by is None)
        if show_group_legend_flag:
            # Create legend elements for groups using the last subplot for positioning
            last_ax = axes.flat[-1] if len(axes.flat) > 0 else axes[0, 0]

            group_legend_elements = []
            for i, group in enumerate(groups):
                group_color = group_colors_list[i]
                group_legend_elements.append(plt.Rectangle((0,0), 1, 1,
                                                         facecolor=group_color, alpha=0.7,
                                                         edgecolor='black', linewidth=0.5,
                                                         label=group))

            # Handle custom legend positions for group legends in faceted plots
            if group_legend_loc == 'below':
                # Place below the entire figure
                group_legend = fig.legend(handles=group_legend_elements, fontsize=group_legend_fontsize,
                                       title=group_by, title_fontsize=group_legend_fontsize,
                                       bbox_to_anchor=(0.5, 0.02), loc='lower center',
                                       ncol=len(groups))
            elif group_legend_loc == 'right':
                # Place to the right of the entire figure
                group_legend = fig.legend(handles=group_legend_elements, fontsize=group_legend_fontsize,
                                       title=group_by, title_fontsize=group_legend_fontsize,
                                       bbox_to_anchor=(0.98, 0.5), loc='center left')
            else:
                # Place in specified location on the last subplot
                group_legend = last_ax.legend(handles=group_legend_elements, loc=group_legend_loc,
                                           fontsize=group_legend_fontsize,
                                           title=group_by, title_fontsize=group_legend_fontsize)

        # Apply y-axis scaling for faceted plots
        # Handle free_y scaling for expression data (left y-axis)
        if not free_y:
            # Calculate global y-limits across all facets
            all_y_mins, all_y_maxs = [], []
            for ax, ax2 in facet_axes_pairs:
                y_min, y_max = ax.get_ylim()
                all_y_mins.append(y_min)
                all_y_maxs.append(y_max)

            if all_y_mins and all_y_maxs:
                global_y_min = min(all_y_mins)
                global_y_max = max(all_y_maxs)

                # Apply global limits to all facets
                for ax, ax2 in facet_axes_pairs:
                    ax.set_ylim(global_y_min, global_y_max)

                print(f"🔗 Applied shared y-axis scaling: ({global_y_min:.3f}, {global_y_max:.3f})")

        # Handle free_mean_y scaling for mean expression data (right y-axis)
        if not free_mean_y:
            # Calculate global mean y-limits across all facets
            all_mean_y_mins, all_mean_y_maxs = [], []
            for ax, ax2 in facet_axes_pairs:
                y_min, y_max = ax2.get_ylim()
                all_mean_y_mins.append(y_min)
                all_mean_y_maxs.append(y_max)

            if all_mean_y_mins and all_mean_y_maxs:
                global_mean_y_min = min(all_mean_y_mins)
                global_mean_y_max = max(all_mean_y_maxs)

                # Apply global limits to all mean axes
                for ax, ax2 in facet_axes_pairs:
                    ax2.set_ylim(global_mean_y_min, global_mean_y_max)

                print(f"🔗 Applied shared mean y-axis scaling: ({global_mean_y_min:.3f}, {global_mean_y_max:.3f})")

        # Apply ylim override if specified (only affects expression/left y-axis)
        if ylim is not None:
            for ax, ax2 in facet_axes_pairs:
                ax.set_ylim(ylim)
            print(f"⚙️  Applied manual y-axis limits: {ylim}")

        # Add overall title
        if title is not None:
            fig.suptitle(title, fontsize=title_fontsize)
        elif gene:
            fig.suptitle(f'{gene} Expression (scVI)', fontsize=title_fontsize)

        plt.tight_layout()
        print(f"✅ Faceted vlnplot_scvi completed for {gene}")
        print(f"📊 Shows: {n_facets} facets by {facet_by}")
        return fig

    else:
        # SINGLE PLOT PATH (original functionality preserved)
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
                    # Calculate true mean of all cells
                    group_mean = group_data.mean()

                    if ax2 is not None:
                        ax2.scatter(i, group_mean, c=mean_color, s=mean_size, marker='o',
                                edgecolors='white', linewidth=1, zorder=10)

                # Add expressing-cells-only mean if enabled
                if plot_mean_pos_frac:
                    if fraction_df is not None:
                        # Use raw data for identifying expressing cells
                        group_indices = plot_data[plot_data['group'] == group].index
                        group_fraction_data = fraction_df.loc[group_indices, 'fraction_expr']
                        expressing_mask = group_fraction_data > fraction_threshold
                        if expressing_mask.sum() > 0:
                            group_mean_pos_frac = group_data[expressing_mask].mean()
                        else:
                            group_mean_pos_frac = 0
                    else:
                        # Fallback: use expression data itself
                        expressing_mask = group_data > expression_threshold
                        if expressing_mask.sum() > 0:
                            group_mean_pos_frac = group_data[expressing_mask].mean()
                        else:
                            group_mean_pos_frac = 0

                    if ax2 is not None and group_mean_pos_frac > 0:
                        ax2.scatter(i, group_mean_pos_frac, c=mean_pos_frac_color, s=mean_pos_frac_size, marker='o',
                                edgecolors='white', linewidth=1, zorder=11)

                    # Collect for axis scaling
                    all_means_pos_frac.append(group_mean_pos_frac)
            else:
                # No data available for this group - add placeholder
                ax.text(i, ax.get_ylim()[1] * 0.5, 'No data', ha='center', va='center',
                        fontsize=14, color='gray', style='italic',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))

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

        # Axis scaling for simple plots
        if plot_mean:
            # Choose which means to use for scaling
            if plot_mean_pos_frac and len(all_means_pos_frac) > 0:
                # Use expressing-cells-only means for scaling (they will be higher)
                scaling_means = [m for m in all_means_pos_frac if m > 0]
                mean_type = "expressing-cells-only"
            else:
                # Use regular true means for scaling (collect them first)
                all_means = []
                for i, group in enumerate(groups):
                    group_data = plot_data[plot_data['group'] == group]['expression']
                    if len(group_data) > 0:
                        all_means.append(group_data.mean())
                scaling_means = [m for m in all_means if m > 0]
                mean_type = "true"

            if len(scaling_means) > 0:
                global_max_mean = max(scaling_means)
                global_max_expression = plot_data['expression'].max()

                if global_max_mean > 0:
                    mean_axis_scale_factor = global_max_expression / global_max_mean
                    mean_axis_max = global_max_expression / mean_axis_scale_factor * 1.1
                    ax2.set_ylim(0, mean_axis_max)
                    print(f"📏 Mean-axis scaling factor ({mean_type} means): {mean_axis_scale_factor:.3f}")
                    print(f"🎯 Mean expression axis max: {mean_axis_max:.3f}")
                else:
                    ax2.set_ylim(0, 1)
                    print(f"⚠️  No positive {mean_type} means found, using default axis")
            else:
                ax2.set_ylim(0, 1)
                print(f"⚠️  No {mean_type} means collected, using default axis")

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
                        # Calculate true mean of all cells for this split
                        split_mean = split_data.mean()

                        if ax2 is not None:
                            ax2.scatter(x_pos, split_mean, c=mean_color, s=mean_size, marker='o',
                                    edgecolors='white', linewidth=1, zorder=10)

                    # Add expressing-cells-only mean if enabled
                    if plot_mean_pos_frac:
                        if fraction_df is not None:
                            # Use raw data for identifying expressing cells
                            split_indices = group_data_all[group_data_all['split'] == split].index
                            split_fraction_data = fraction_df.loc[split_indices, 'fraction_expr']
                            expressing_mask = split_fraction_data > fraction_threshold
                            if expressing_mask.sum() > 0:
                                split_mean_pos_frac = split_data[expressing_mask].mean()
                            else:
                                split_mean_pos_frac = 0
                        else:
                            # Fallback: use expression data itself
                            expressing_mask = split_data > expression_threshold
                            if expressing_mask.sum() > 0:
                                split_mean_pos_frac = split_data[expressing_mask].mean()
                            else:
                                split_mean_pos_frac = 0

                        if ax2 is not None and split_mean_pos_frac > 0:
                            ax2.scatter(x_pos, split_mean_pos_frac, c=mean_pos_frac_color, s=mean_pos_frac_size, marker='o',
                                    edgecolors='white', linewidth=1, zorder=11)

                        # Collect for axis scaling
                        all_means_pos_frac.append(split_mean_pos_frac)
                else:
                    # No data available for this group + split combination - add placeholder
                    x_offset = (j - (len(splits) - 1) / 2) * violin_width
                    x_pos = i + x_offset
                    ax.text(x_pos, ax.get_ylim()[1] * 0.5, 'No data', ha='center', va='center',
                            fontsize=12, color='gray', style='italic',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.3))

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
        # If this is the only legend being shown, use legend_loc instead of group_legend_loc
        effective_group_loc = legend_loc if not show_split_legend else group_legend_loc

        if effective_group_loc == 'below':
            # Adjust position if split legend is also below
            y_offset = -0.25 if not (show_split_legend and legend_loc == 'below') else -0.35
            group_legend = ax.legend(handles=group_legend_elements, fontsize=group_legend_fontsize,
                                   title=group_by, title_fontsize=group_legend_fontsize,
                                   bbox_to_anchor=(0.5, y_offset), loc='center',
                                   ncol=len(groups))
        elif effective_group_loc == 'right':
            # Adjust position if split legend is also right
            x_offset = 1.15 if not (show_split_legend and legend_loc == 'right') else 1.30
            group_legend = ax.legend(handles=group_legend_elements, fontsize=group_legend_fontsize,
                                   title=group_by, title_fontsize=group_legend_fontsize,
                                   bbox_to_anchor=(x_offset, 0.5), loc='center left')
        else:
            group_legend = ax.legend(handles=group_legend_elements, loc=effective_group_loc,
                                   fontsize=group_legend_fontsize,
                                   title=group_by, title_fontsize=group_legend_fontsize)

        # If we have both legends, we need to add the group legend manually
        if show_split_legend:
            ax.add_artist(split_legend)  # Keep the split legend

    # Format axes
    ax.set_xticks(range(len(groups)))

    # Determine x-axis labels (use abbreviations if provided)
    if group_labels is not None:
        # Use abbreviated labels when group_labels is provided
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

    # Apply axis tick font sizes
    ax.tick_params(axis='y', labelsize=axis_tick_fontsize)
    ax.tick_params(axis='x', labelsize=xlabel_fontsize)

    if plot_mean:
        ax2.set_ylabel('Mean Expression', fontsize=ylabel_mean_fontsize, color=mean_color)
        ax2.tick_params(axis='y', labelcolor=mean_color, labelsize=axis_tick_fontsize)

    # Set title
    plot_title = title or f'{gene} Expression (scVI)'
    ax.set_title(plot_title, fontsize=title_fontsize)


    # Add annotations for new comparison system
    if comparison_stats:
        print(f"📊 Adding comparison annotations...")
        y_min, y_max = ax.get_ylim()

        # Calculate annotation heights
        annotation_height_start = y_max * 1.05

        annotation_height_step = y_max * 0.08
        comparison_annotation_count = 0

        for comp_key, stats in comparison_stats.items():
            proba_de = stats.get('proba_de', 0)
            lfc = stats.get('lfc_mean', 0)
            bayes_factor = stats.get('bayes_factor', 0)

            # Convert Series to scalar if needed
            if hasattr(proba_de, 'item'):
                proba_de = proba_de.item()
            if hasattr(lfc, 'item'):
                lfc = lfc.item()
            if hasattr(bayes_factor, 'item'):
                bayes_factor = bayes_factor.item()

            # Determine significance level
            if proba_de >= proba_de_thresholds[2]:  # 95%
                stars = "***"
            elif proba_de >= proba_de_thresholds[1]:  # 80%
                stars = "**"
            elif proba_de >= proba_de_thresholds[0]:  # 60%
                stars = "*"
            else:
                stars = "ns"

            # Always add visual annotations for all comparisons (including ns)
            if True:
                print(f"   ✅ {comp_key}: {stars} (P={proba_de:.3f}, LFC={lfc:.2f})")

                # Parse comparison key to get positions
                if comp_key.startswith('group_'):
                    # Group comparison: extract group names
                    parts = comp_key.replace('group_', '').split('_vs_')
                    if len(parts) == 2:
                        group1, group2 = parts
                        try:
                            x1 = groups.index(group1)
                            x2 = groups.index(group2)
                            y_pos = annotation_height_start + comparison_annotation_count * annotation_height_step

                            # Add significance line
                            ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1)
                            ax.plot([x1, x1], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)
                            ax.plot([x2, x2], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)

                            # Add annotation text with flexible detail levels
                            x_center = (x1 + x2) / 2

                            # Create flexible annotation based on detailed_stats parameter
                            if detailed_stats == "full":
                                # Show everything: stars + Prob(DE) + LFC + BF
                                if stars == "ns":
                                    annotation = f"ns\nP(DE)={proba_de:.2f}\nLFC={lfc:.1f}\nBF={bayes_factor:.1f}"
                                else:
                                    annotation = f"{stars}\nP(DE)={proba_de:.2f}\nLFC={lfc:.1f}\nBF={bayes_factor:.1f}"
                            elif detailed_stats == "medium":
                                # Show stars + Prob(DE) + LFC
                                if stars == "ns":
                                    annotation = f"ns\nP(DE)={proba_de:.2f}\nLFC={lfc:.1f}"
                                else:
                                    annotation = f"{stars}\nP(DE)={proba_de:.2f}\nLFC={lfc:.1f}"
                            elif detailed_stats == "minimal":
                                # Show stars + Prob(DE) only
                                if stars == "ns":
                                    annotation = f"ns\nP(DE)={proba_de:.2f}"
                                else:
                                    annotation = f"{stars}\nP(DE)={proba_de:.2f}"
                            elif detailed_stats:  # True or backward compatibility
                                # Original behavior: stars + Prob(DE) + LFC (if >= 0.1)
                                if stars == "ns":
                                    annotation = f"ns\nP(DE)={proba_de:.2f}"
                                else:
                                    annotation = f"{stars}\nP(DE)={proba_de:.2f}"
                                if abs(lfc) >= 0.1:
                                    annotation += f"\nLFC={lfc:.1f}"
                            else:
                                # Just stars
                                annotation = stars

                            ax.text(x_center, y_pos + y_max*0.03, annotation, ha='center', va='bottom',
                                   fontsize=14, fontweight='bold')
                            comparison_annotation_count += 1
                        except ValueError:
                            print(f"   ⚠️ Groups {group1} or {group2} not found in plot")

                elif comp_key.startswith('split_'):
                    # Split comparison: handle within-group positioning
                    parts = comp_key.replace('split_', '').split('_')
                    if len(parts) >= 3:
                        within_group = parts[0]
                        split1, split2 = parts[1], parts[2]
                        try:
                            i = groups.index(within_group)

                            # Calculate x positions for the two split categories within this group
                            x1 = i - 0.2  # Position of first split category
                            x2 = i + 0.2  # Position of second split category

                            # Position split annotations well within the data area for clear visibility
                            y_pos = y_max * 0.85

                            # Add significance line
                            ax.plot([x1, x2], [y_pos, y_pos], 'k-', linewidth=1)
                            ax.plot([x1, x1], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)
                            ax.plot([x2, x2], [y_pos - y_max*0.01, y_pos + y_max*0.01], 'k-', linewidth=1)

                            # Add annotation text with flexible detail levels
                            x_center = (x1 + x2) / 2

                            # Create flexible annotation based on detailed_stats parameter
                            if detailed_stats == "full":
                                # Show everything: stars + Prob(DE) + LFC + BF
                                if stars == "ns":
                                    annotation = f"ns\nP(DE)={proba_de:.2f}\nLFC={lfc:.1f}\nBF={bayes_factor:.1f}"
                                else:
                                    annotation = f"{stars}\nP(DE)={proba_de:.2f}\nLFC={lfc:.1f}\nBF={bayes_factor:.1f}"
                            elif detailed_stats == "medium":
                                # Show stars + Prob(DE) + LFC
                                if stars == "ns":
                                    annotation = f"ns\nP(DE)={proba_de:.2f}\nLFC={lfc:.1f}"
                                else:
                                    annotation = f"{stars}\nP(DE)={proba_de:.2f}\nLFC={lfc:.1f}"
                            elif detailed_stats == "minimal":
                                # Show stars + Prob(DE) only
                                if stars == "ns":
                                    annotation = f"ns\nP(DE)={proba_de:.2f}"
                                else:
                                    annotation = f"{stars}\nP(DE)={proba_de:.2f}"
                            elif detailed_stats:  # True or backward compatibility
                                # Original behavior: stars + Prob(DE) + LFC (if >= 0.1)
                                if stars == "ns":
                                    annotation = f"ns\nP(DE)={proba_de:.2f}"
                                else:
                                    annotation = f"{stars}\nP(DE)={proba_de:.2f}"
                                if abs(lfc) >= 0.1:
                                    annotation += f"\nLFC={lfc:.1f}"
                            else:
                                # Just stars
                                annotation = stars

                            ax.text(x_center, y_pos + y_max*0.03, annotation, ha='center', va='bottom',
                                   fontsize=14, fontweight='bold')
                        except ValueError:
                            print(f"   ⚠️ Group {within_group} not found in plot")

        # Adjust y-axis to accommodate annotations
        total_annotations = comparison_annotation_count
        if total_annotations > 0:
            # For split comparisons, we need extra space at 1.05 * y_max + text height
            # For group comparisons, we use the standard annotation height calculation
            max_annotation_height = max(
                annotation_height_start + total_annotations * annotation_height_step,
                y_max * 1.25  # Allow ample space for split comparison text at 1.05 * y_max
            )
            new_y_max = max_annotation_height + y_max * 0.1
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