import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_log_data(log_file_path="logs/batch_testing.log"):
  try:
    df = pd.read_csv(log_file_path)
    df['block_size'] = df['block_size'].str.replace('block_', '').astype(int)
    df[['rows_a', 'cols_a']] = df['matrix_a_shape'].str.split('x', expand=True).astype(int)
    df[['rows_b', 'cols_b']] = df['matrix_b_shape'].str.split('x', expand=True).astype(int)
    df['matrix_type'] = df.apply(lambda x: 'Square' if x['rows_a'] == x['cols_a'] == x['rows_b'] == x['cols_b'] else 'Rectangular', axis=1)
    df['matrix_size'] = df['rows_a']
    return df
  except FileNotFoundError:
    print(f"Error: {log_file_path} not found. Please run the C test first.")
    return None

def create_visualizations(df):
  plt.style.use('seaborn-v0_8')
  colors = {'hybrid_parallel': '#2E86AB', 'hybrid_transposed': '#A23B72'}

  fig = plt.figure(figsize=(20, 24))

  square_df = df[df['matrix_type'] == 'Square']
  rect_df = df[df['matrix_type'] == 'Rectangular']

  matrix_sizes = sorted(df['matrix_size'].unique())

  plot_idx = 1

  for i, size in enumerate(matrix_sizes):
    square_data = square_df[square_df['matrix_size'] == size]
    rect_data = rect_df[rect_df['matrix_size'] == size]
    
    plt.subplot(5, 4, plot_idx)
    for method in ['hybrid_parallel', 'hybrid_transposed']:
      method_data = square_data[square_data['method'] == method]
      plt.plot(method_data['block_size'], method_data['gflops'], marker='o', linewidth=2.5, markersize=6, color=colors[method], label=method.replace('_', ' ').title())
    plt.title(f'Square {size}x{size} - GFLOPS vs Block Size', fontweight='bold')
    plt.xlabel('Block Size')
    plt.ylabel('GFLOPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)

    plt.subplot(5, 4, plot_idx + 1)
    for method in ['hybrid_parallel', 'hybrid_transposed']:
      method_data = square_data[square_data['method'] == method]
      plt.plot(method_data['block_size'], method_data['time_seconds'] * 1000, marker='s', linewidth=2.5, markersize=6, color=colors[method], label=method.replace('_', ' ').title())
    plt.title(f'Square {size}x{size} - Runtime vs Block Size', fontweight='bold')
    plt.xlabel('Block Size')
    plt.ylabel('Runtime (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')

    plt.subplot(5, 4, plot_idx + 2)
    for method in ['hybrid_parallel', 'hybrid_transposed']:
      method_data = rect_data[rect_data['method'] == method]
      plt.plot(method_data['block_size'], method_data['gflops'], marker='o', linewidth=2.5, markersize=6, color=colors[method], label=method.replace('_', ' ').title())
    plt.title(f'Rectangular {size}x{size//2}@{size//2}x{size} - GFLOPS', fontweight='bold')
    plt.xlabel('Block Size')
    plt.ylabel('GFLOPS')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)

    plt.subplot(5, 4, plot_idx + 3)
    for method in ['hybrid_parallel', 'hybrid_transposed']:
      method_data = rect_data[rect_data['method'] == method]
      plt.plot(method_data['block_size'], method_data['time_seconds'] * 1000, marker='s', linewidth=2.5, markersize=6, color=colors[method], label=method.replace('_', ' ').title())
    plt.title(f'Rectangular {size}x{size//2}@{size//2}x{size} - Runtime', fontweight='bold')
    plt.xlabel('Block Size')
    plt.ylabel('Runtime (ms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')

    plot_idx += 4

  plt.tight_layout(pad=3.0)
  plt.savefig('media/performance_analysis.png', dpi=300, bbox_inches='tight')
  plt.show()

def create_summary_plots(df):
  fig, axes = plt.subplots(2, 2, figsize=(16, 12))

  square_df = df[df['matrix_type'] == 'Square']
  rect_df = df[df['matrix_type'] == 'Rectangular']

  pivot_gflops_sq = square_df.pivot_table(values='gflops', index='block_size', columns=['method', 'matrix_size'], aggfunc='mean')
  pivot_time_sq = square_df.pivot_table(values='time_seconds', index='block_size', columns=['method', 'matrix_size'], aggfunc='mean')

  im1 = axes[0,0].imshow(pivot_gflops_sq.T.values, aspect='auto', cmap='viridis')
  axes[0,0].set_title('Square Matrices - GFLOPS Heatmap', fontweight='bold')
  axes[0,0].set_xlabel('Block Size Index')
  axes[0,0].set_ylabel('Method & Matrix Size')
  axes[0,0].set_xticks(range(len(pivot_gflops_sq.index)))
  axes[0,0].set_xticklabels(pivot_gflops_sq.index)
  axes[0,0].set_yticks(range(len(pivot_gflops_sq.columns)))
  axes[0,0].set_yticklabels([f"{col[0]}_{col[1]}" for col in pivot_gflops_sq.columns], rotation=45)
  plt.colorbar(im1, ax=axes[0,0], label='GFLOPS')

  im2 = axes[0,1].imshow(np.log10(pivot_time_sq.T.values * 1000), aspect='auto', cmap='plasma')
  axes[0,1].set_title('Square Matrices - Runtime Heatmap (log10 ms)', fontweight='bold')
  axes[0,1].set_xlabel('Block Size Index')
  axes[0,1].set_ylabel('Method & Matrix Size')
  axes[0,1].set_xticks(range(len(pivot_time_sq.index)))
  axes[0,1].set_xticklabels(pivot_time_sq.index)
  axes[0,1].set_yticks(range(len(pivot_time_sq.columns)))
  axes[0,1].set_yticklabels([f"{col[0]}_{col[1]}" for col in pivot_time_sq.columns], rotation=45)
  plt.colorbar(im2, ax=axes[0,1], label='log10(Runtime ms)')

  method_comparison = df.groupby(['method', 'matrix_size', 'matrix_type']).agg({'gflops': 'max', 'time_seconds': 'min'}).reset_index()
  square_comp = method_comparison[method_comparison['matrix_type'] == 'Square']
  rect_comp = method_comparison[method_comparison['matrix_type'] == 'Rectangular']

  width = 0.35
  x_sq = np.arange(len(square_comp['matrix_size'].unique()))
  parallel_gflops_sq = square_comp[square_comp['method'] == 'hybrid_parallel']['gflops'].values
  transposed_gflops_sq = square_comp[square_comp['method'] == 'hybrid_transposed']['gflops'].values

  axes[1,0].bar(x_sq - width/2, parallel_gflops_sq, width, label='Hybrid Parallel', color='#2E86AB', alpha=0.8)
  axes[1,0].bar(x_sq + width/2, transposed_gflops_sq, width, label='Hybrid Transposed', color='#A23B72', alpha=0.8)
  axes[1,0].set_title('Best GFLOPS Comparison - Square Matrices', fontweight='bold')
  axes[1,0].set_xlabel('Matrix Size')
  axes[1,0].set_ylabel('Max GFLOPS')
  axes[1,0].set_xticks(x_sq)
  axes[1,0].set_xticklabels(square_comp['matrix_size'].unique())
  axes[1,0].legend()
  axes[1,0].grid(True, alpha=0.3)

  x_rect = np.arange(len(rect_comp['matrix_size'].unique()))
  parallel_gflops_rect = rect_comp[rect_comp['method'] == 'hybrid_parallel']['gflops'].values
  transposed_gflops_rect = rect_comp[rect_comp['method'] == 'hybrid_transposed']['gflops'].values

  axes[1,1].bar(x_rect - width/2, parallel_gflops_rect, width, label='Hybrid Parallel', color='#2E86AB', alpha=0.8)
  axes[1,1].bar(x_rect + width/2, transposed_gflops_rect, width, label='Hybrid Transposed', color='#A23B72', alpha=0.8)
  axes[1,1].set_title('Best GFLOPS Comparison - Rectangular Matrices', fontweight='bold')
  axes[1,1].set_xlabel('Matrix Size')
  axes[1,1].set_ylabel('Max GFLOPS')
  axes[1,1].set_xticks(x_rect)
  axes[1,1].set_xticklabels(rect_comp['matrix_size'].unique())
  axes[1,1].legend()
  axes[1,1].grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('media/summary_analysis.png', dpi=300, bbox_inches='tight')
  plt.show()

def create_block_size_analysis(df):
  fig, axes = plt.subplots(2, 3, figsize=(18, 12))

  block_sizes = sorted(df['block_size'].unique())
  methods = ['hybrid_parallel', 'hybrid_transposed']

  for i, matrix_type in enumerate(['Square', 'Rectangular']):
    type_df = df[df['matrix_type'] == matrix_type]

    for j, metric in enumerate(['gflops', 'time_seconds']):
      ax = axes[i, j]

      for method in methods:
        method_df = type_df[type_df['method'] == method]
        avg_by_block = method_df.groupby('block_size')[metric].mean()

        color = '#2E86AB' if method == 'hybrid_parallel' else '#A23B72'
        ax.plot(avg_by_block.index, avg_by_block.values if metric == 'gflops' else avg_by_block.values * 1000,marker='o', linewidth=3, markersize=8, color=color, label=method.replace('_', ' ').title())

      ax.set_title(f'{matrix_type} - {"GFLOPS" if metric == "gflops" else "Runtime"} by Block Size', fontweight='bold', fontsize=12)
      ax.set_xlabel('Block Size')
      ax.set_ylabel('GFLOPS' if metric == 'gflops' else 'Runtime (ms)')
      ax.legend()
      ax.grid(True, alpha=0.3)
      ax.set_xscale('log', base=2)
      if metric == 'time_seconds': ax.set_yscale('log')

    type_df_best = type_df.loc[type_df.groupby(['method', 'matrix_size'])['gflops'].idxmax()]

    ax = axes[i, 2]
    for method in methods:
      method_best = type_df_best[type_df_best['method'] == method]
      color = '#2E86AB' if method == 'hybrid_parallel' else '#A23B72'
      ax.scatter(method_best['matrix_size'], method_best['block_size'], s=100, alpha=0.7, color=color, label=method.replace('_', ' ').title())

    ax.set_title(f'{matrix_type} - Optimal Block Size by Matrix Size', fontweight='bold', fontsize=12)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Optimal Block Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log', base=2)

  plt.tight_layout()
  plt.savefig('media/block_size_analysis.png', dpi=300, bbox_inches='tight')
  plt.show()

def create_efficiency_analysis(df):
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  efficiency_data = []
  for _, row in df.iterrows():
    theoretical_peak = 100.0
    efficiency = (row['gflops'] / theoretical_peak) * 100
    efficiency_data.append({
      'method': row['method'],
      'matrix_type': row['matrix_type'],
      'matrix_size': row['matrix_size'],
      'block_size': row['block_size'],
      'efficiency_percent': min(efficiency, 100)
    })

  eff_df = pd.DataFrame(efficiency_data)

  for i, matrix_type in enumerate(['Square', 'Rectangular']):
    ax = axes[i]
    type_data = eff_df[eff_df['matrix_type'] == matrix_type]

    sns.boxplot(data=type_data, x='method', y='efficiency_percent', ax=ax, palette=['#2E86AB', '#A23B72'])
    ax.set_title(f'{matrix_type} Matrices - Efficiency Distribution', fontweight='bold')
    ax.set_xlabel('Method')
    ax.set_ylabel('Efficiency (%)')
    ax.set_xticklabels(['Hybrid Parallel', 'Hybrid Transposed'])
    ax.grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig('media/efficiency_analysis.png', dpi=300, bbox_inches='tight')
  plt.show()

def print_performance_summary(df):
  print("=== PERFORMANCE SUMMARY ===")
  print()

  for matrix_type in ['Square', 'Rectangular']:
    print(f"{matrix_type} Matrices:")
    type_df = df[df['matrix_type'] == matrix_type]

    best_overall = type_df.loc[type_df['gflops'].idxmax()]
    print(f"  Best Overall: {best_overall['method']} with {best_overall['gflops']:.2f} GFLOPS")
    print(f"    Matrix: {best_overall['matrix_a_shape']}, Block: {best_overall['block_size']}")

    for method in ['hybrid_parallel', 'hybrid_transposed']:
      method_df = type_df[type_df['method'] == method]
      best_method = method_df.loc[method_df['gflops'].idxmax()]
      avg_gflops = method_df['gflops'].mean()
      print(f"  {method.replace('_', ' ').title()}:")
      print(f"    Best: {best_method['gflops']:.2f} GFLOPS (block={best_method['block_size']})")
      print(f"    Average: {avg_gflops:.2f} GFLOPS")
    print()

  print("=== BLOCK SIZE ANALYSIS ===")
  optimal_blocks = df.loc[df.groupby(['method', 'matrix_type', 'matrix_size'])['gflops'].idxmax()]
  optimal_summary = optimal_blocks.groupby(['method', 'matrix_type'])['block_size'].agg(['mean', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]]).round(1)
  optimal_summary.columns = ['mean', 'most_common']
  print(optimal_summary)
  print()

def main():
  print("Matrix Multiplication Performance Analyzer")
  print("==========================================")
  
  df = load_log_data()
  if df is None: return
  print(f"Loaded {len(df)} test results")
  print(f"Methods tested: {', '.join(df['method'].unique())}")
  print(f"Block sizes: {sorted(df['block_size'].unique())}")
  print(f"Matrix sizes: {sorted(df['matrix_size'].unique())}")
  print()

  Path("logs").mkdir(exist_ok=True)
  print("Creating detailed performance visualizations...")
  create_visualizations(df)
  print("Creating summary analysis...")
  create_summary_plots(df)
  print("Creating block size analysis...")
  create_block_size_analysis(df)
  print("Creating efficiency analysis...")
  create_efficiency_analysis(df)
  print_performance_summary(df)
  print("Analysis complete! Check media/ directory for visualizations.")

if __name__ == "__main__":
  main()