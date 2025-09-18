"""
Generic helper functions for computing split effects within groups for vlnplot_scvi.
"""

import numpy as np
import pandas as pd


def compute_split_effects_within_groups(adata, gene, group_by, split_by,
                                       split_categories, scvi_model, mode='change'):
    """
    Compute split_by effects within each group_by category.

    This is a generic function that works for any group_by/split_by combination:
    - group_by='condition', split_by='treatment' → treatment effects within each condition
    - group_by='celltype', split_by='treatment' → treatment effects within each celltype
    - group_by='timepoint', split_by='genotype' → genotype effects within each timepoint

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    gene : str
        Gene name to analyze
    group_by : str
        Column name for grouping (e.g., 'condition', 'celltype', 'timepoint')
    split_by : str
        Column name for splitting/comparison (e.g., 'treatment', 'genotype', 'stimulus')
    split_categories : list of str
        Two categories to compare from split_by column (e.g., ['dupi', 'pre'])
    scvi_model : scvi model
        Trained scVI model for differential expression
    mode : str, default 'change'
        scVI DE mode ('change' or 'vanilla')

    Returns
    -------
    dict
        Dictionary with group_by categories as keys, each containing:
        {'proba_de': float, 'lfc_mean': float}

    Examples
    --------
    # Treatment effects within conditions
    treatment_by_condition = compute_split_effects_within_groups(
        adata, 'GNLY', 'condition', 'treatment', ['dupi', 'pre'], model
    )

    # Genotype effects within celltypes
    genotype_by_celltype = compute_split_effects_within_groups(
        adata, 'GNLY', 'celltype', 'genotype', ['WT', 'KO'], model
    )
    """
    split_effects_by_group = {}

    split1, split2 = split_categories

    for group in adata.obs[group_by].unique():
        # Get cells from this group only
        group_mask = adata.obs[group_by] == group
        adata_group = adata[group_mask].copy()

        # Check if we have both split categories in this group
        split_counts = adata_group.obs[split_by].value_counts()
        if (split1 in split_counts and split2 in split_counts and
            split_counts[split1] >= 10 and split_counts[split2] >= 10):

            print(f"Computing {split1} vs {split2} within {group} "
                  f"({split_counts[split1]} vs {split_counts[split2]} cells)")

            # Compute DE for this group
            de_group = scvi_model.differential_expression(
                adata_group,
                groupby=split_by,
                mode=mode
            )

            # Extract gene stats for this group (handle 1-vs-all results)
            if gene in de_group.index:
                gene_stats = de_group.loc[gene]

                # Extract the two comparisons and calculate pairwise
                split1_vs_rest = gene_stats[gene_stats['group1'] == split1].iloc[0]
                split2_vs_rest = gene_stats[gene_stats['group1'] == split2].iloc[0]

                # Calculate pairwise LFC: log2(split1/split2)
                scale1 = split1_vs_rest['scale1']
                scale2 = split2_vs_rest['scale1']

                if scale2 > 0 and scale1 > 0:
                    lfc_pairwise = np.log2(scale1 / scale2)
                else:
                    lfc_pairwise = 0

                proba_de_val = split1_vs_rest['proba_de']

                split_effects_by_group[group] = {
                    'proba_de': proba_de_val,
                    'lfc_mean': lfc_pairwise
                }
                print(f"  {group}: P(DE)={proba_de_val:.3f}, LFC={lfc_pairwise:.2f}")
            else:
                print(f"  {group}: {gene} not found in DE results")
        else:
            print(f"  {group}: Insufficient cells "
                  f"({split1}: {split_counts.get(split1, 0)}, "
                  f"{split2}: {split_counts.get(split2, 0)})")

    return split_effects_by_group


def plot_split_effects_within_groups(adata, gene, group_by, split_by, split_categories,
                                    split_stats, **plot_kwargs):
    """
    Convenience function for Pattern A: Show split_by effects within group_by categories.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    gene : str
        Gene to plot
    group_by : str
        Grouping variable (e.g., 'condition', 'celltype')
    split_by : str
        Splitting variable (e.g., 'treatment', 'genotype')
    split_categories : list
        Two categories to compare (e.g., ['dupi', 'pre'])
    split_stats : dict
        Pre-computed split statistics from compute_split_effects_within_groups
    **plot_kwargs
        Additional arguments passed to vlnplot_scvi

    Returns
    -------
    matplotlib figure
    """
    from .violin import vlnplot_scvi

    return vlnplot_scvi(
        adata,
        gene=gene,
        group_by=group_by,
        split_by=split_by,
        split_effects=[tuple(split_categories)],
        split_stats=split_stats,
        **plot_kwargs
    )


def plot_pure_group_comparisons(adata, gene, group_by, subset_by=None,
                               group_effects=None, **plot_kwargs):
    """
    Convenience function for Pattern B1/B2: Pure group comparisons after subsetting.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    gene : str
        Gene to plot
    group_by : str
        Grouping variable for comparison
    subset_by : tuple, optional
        (column, value) to subset data first (e.g., ('treatment', 'dupi'))
    group_effects : list of tuples
        Group comparisons to perform
    **plot_kwargs
        Additional arguments passed to vlnplot_scvi

    Returns
    -------
    matplotlib figure
    """
    from .violin import vlnplot_scvi

    if subset_by:
        adata_sub = adata[adata.obs[subset_by[0]] == subset_by[1]].copy()
    else:
        adata_sub = adata

    return vlnplot_scvi(
        adata_sub,
        gene=gene,
        group_by=group_by,
        group_effects=group_effects,
        **plot_kwargs
    )