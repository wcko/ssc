# ssc: Signaling Single Cell - Visualizations

A Python package for creating publication-ready visualizations of single-cell RNA sequencing data.

## Installation

### Development Installation (Recommended during active development)

```bash
# Clone the repository
git clone https://github.com/yourusername/ssc.git
cd ssc

# Install in development mode
pip install -e .
```

### PyPI Installation (Coming soon)

```bash
pip install ssc
```

## Quick Start

```python
import ssc
import scanpy as sc

# Load your AnnData object
adata = sc.read_h5ad('your_data.h5ad')

# Basic violin plot
fig = ssc.vlnplot(adata, 'GNLY', 'cell_type')

# Split violin with custom colors
fig = ssc.vlnplot(adata, 'GNLY', 'condition', 
                  split_by='treatment',
                  split_colors={'pre': 'lightblue', 'dupi': 'darkblue'})

# Faceted plot with custom ordering
fig = ssc.vlnplot(adata, 'GNLY', 'condition', 
                  facet_by='subject', facet_col='cell_type',
                  group_order=['Nonlesional', 'SADBE', 'Metal'])
```

## Features

- **Publication-ready plots**: High-quality violin plots with comprehensive customization
- **Split violin plots**: Compare conditions within each group
- **Faceted plots**: Create subplot grids by multiple variables
- **Custom colors**: Dictionary-based color mapping for groups and splits
- **R-style scaling**: Dual y-axis with proper mean expression scaling
- **Custom ordering**: Control category order for all grouping variables
- **Rich statistics**: Cell counts, fraction expressing, and mean expression
- **Layer support**: Plot from any AnnData layer or raw counts

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scanpy >= 1.8.0

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License