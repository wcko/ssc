"""
Violin plot functions for single-cell RNA sequencing data visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse


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


def vlnplot(adata, gene, group_by, 
                    title=None, 
                    layer=None,
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
                    show_stats=True,
                    number_labels=True,
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
                    mean_color='black',
                    free_y=True,
                    free_mean_y=False):
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
        Column name in adata.obs for split violin plots
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
    show_stats : bool, default True
        Whether to show cell statistics
    number_labels : bool, default True
        Whether to show cell count and fraction expressing below x-axis
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
    mean_color : str, default 'black'
        Color for mean expression dots and axis
    free_y : bool, default True
        If True, allow independent gene expression y-axis scaling for each subplot
    free_mean_y : bool, default False
        If True, allow independent mean expression y-axis scaling for each subplot
    
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
    """

    # Extract and validate data
    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene} not found in dataset")

    if layer is not None:
        if layer not in adata.layers.keys():
            raise ValueError(f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}")
        expression_data = adata[:, gene].layers[layer].toarray().flatten()
        data_source = f"layer '{layer}'"
    else:
        expression_data = adata[:, gene].X.toarray().flatten()
        data_source = "adata.X (default)"

    # Build plot data
    plot_data = pd.DataFrame({
        'group': adata.obs[group_by].values,
        'expression': expression_data
    })

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
    print(f"📋 Number labels: {number_labels} (font size: {number_fontsize})")
    print(f"🏷️  X-axis labels: size {xlabel_fontsize}, rotation {xlabel_rotation}°, align {xlabel_ha}")

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
            n_cols = len(facet_categories)
        else:
            facet_categories = [None]
            n_cols = 1

        if facet_col is not None:
            facet_col_categories = _get_category_order(plot_data['facet_col'], facet_col_order, 'Facet columns')
            n_rows = len(facet_col_categories)
        else:
            facet_col_categories = [None]
            n_rows = 1

        # Better default figure size
        if facet_figsize is not None:
            figsize = facet_figsize
        else:
            figsize = (6 * n_cols, 5 * n_rows)  # Clean proportions

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

    # Calculate R-style scaling for means
    if plot_mean:
        grouping_columns = ['group']
        if facet_by is not None:
            grouping_columns.append('facet_by')
        if facet_col is not None:
            grouping_columns.append('facet_col')
        if split_by is not None:
            grouping_columns.append('split')

        print(f"🔍 Using grouping columns: {grouping_columns}")
        group_means = plot_data.groupby(grouping_columns, observed=True)['expression'].mean()
        global_max_mean = group_means.max() if len(group_means) > 0 else 1.0
        global_max_expression = plot_data['expression'].max()

        if global_max_mean > 0:
            r_scale_factor = global_max_expression / global_max_mean
            print(f"📏 R-style scaling factor: {r_scale_factor:.3f}")
        else:
            r_scale_factor = 1.0
            print(f"⚠️  No positive means found, using scale factor: 1.0")

    # Storage for scaling
    all_means = []
    subplot_y_data = {}
    subplot_mean_data = {}

    # Plot each subplot
    for idx, (ax1, ax2, row, col) in enumerate(ax_pairs):

        # Determine which data subset to use
        if is_faceted:
            facet_col_cat = facet_col_categories[row] if facet_col_categories[0] is not None else None
            facet_cat = facet_categories[col] if facet_categories[0] is not None else None

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

                    group_mean = group_expr.mean()
                    subplot_means.append(group_mean)

                    if plot_mean:
                        ax2.scatter(i, group_mean, c=mean_color, s=60, marker='o',
                                edgecolors='white', linewidth=1, zorder=10)
                else:
                    _draw_empty_placeholder_clean(ax1, i, "No data")
                    subplot_means.append(0)

            # Cell statistics positioning - closer to x-axis
            if number_labels:
                for i, group in enumerate(groups):
                    group_data = subset_data[subset_data['group'] == group]['expression']
                    if len(group_data) > 0:
                        n_cells = len(group_data)
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
                            mean_positions.append(x_pos)
                            mean_values.append(split_data.mean())
                            subplot_means.append(split_data.mean())
                    else:
                        _draw_empty_split_placeholder(ax1, x_pos, split_width * 0.8, split_colors_palette[j])

                if not group_has_data:
                    _draw_empty_placeholder_clean(ax1, i, "No data")

            if plot_mean:
                for pos, mean_val in zip(mean_positions, mean_values):
                    ax2.scatter(pos, mean_val, c=mean_color, s=60, marker='o',
                            edgecolors='white', linewidth=1, zorder=10)

            # Cell statistics for split violins - positioned closer to x-axis
            if number_labels:
                for i, group in enumerate(groups):
                    group_data = subset_data[subset_data['group'] == group]

                    for j, split_cat in enumerate(global_splits):
                        split_data = group_data[group_data['split'] == split_cat]['expression'].values
                        x_pos = i - (violin_width/2) + (j + 0.5) * split_width

                        if len(split_data) > 0:
                            n_cells = len(split_data)
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

    # Proper free_mean_y implementation
    if plot_mean:
        if free_mean_y:
            # Independent R-style scaling for each subplot's mean axis
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
            # Global R-style scaling for all mean axes
            if r_scale_factor > 0:
                mean_axis_max = global_max_expression / r_scale_factor * 1.1
                for _, ax2, _, _ in ax_pairs:
                    ax2.set_ylim(0, mean_axis_max)
                print(f"🎯 Global mean expression axis max: {mean_axis_max:.3f} (R-style scaling)")
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
    if number_labels:
        plt.subplots_adjust(left=0.15, bottom=0.35, top=0.92, right=0.95, hspace=0.3)
    else:
        plt.subplots_adjust(left=0.15, bottom=0.2, top=0.92, right=0.95, hspace=0.3)

    plt.tight_layout()

    return fig