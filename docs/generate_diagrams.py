"""
Generate architecture diagrams and model comparison charts for the README.
No raw data needed -- these are structural/informational visuals.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.grid': False,
    'text.color': '#222222',
})


def draw_pipeline_diagram():
    """Draw the full data pipeline and model architecture."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Color scheme
    c_data = '#4A90D9'
    c_transform = '#7B68EE'
    c_stat = '#2ECC71'
    c_ml = '#E67E22'
    c_neural = '#E74C3C'
    c_eval = '#1ABC9C'
    c_arrow = '#555555'

    box_style = dict(boxstyle='round,pad=0.4', linewidth=1.5)

    def draw_box(x, y, w, h, text, color, fontsize=10, bold=False):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.15',
            facecolor=color, edgecolor='#333333',
            linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w / 2, y + h / 2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight=weight, color='white')

    def draw_arrow(x1, y1, x2, y2, style='->', color=c_arrow):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                    lw=1.8, connectionstyle='arc3,rad=0'))

    # Title
    ax.text(7, 6.7, 'Forecasting Pipeline Architecture', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#222222')

    # Row 1: Data ingestion
    draw_box(0.3, 5.5, 2.5, 0.8, 'Raw Order Data\n(30-min intervals)', c_data, 10, True)
    draw_arrow(2.8, 5.9, 3.5, 5.9)

    draw_box(3.5, 5.5, 2.8, 0.8, 'EDA & Diagnostics\nACF / PSD / MSTL', c_data, 10)
    draw_arrow(6.3, 5.9, 7.0, 5.9)

    draw_box(7.0, 5.5, 2.8, 0.8, 'Hampel Filter\nOutlier Removal', c_transform, 10)
    draw_arrow(9.8, 5.9, 10.5, 5.9)

    draw_box(10.5, 5.5, 3.0, 0.8, 'Log Transform\n+ Differencing', c_transform, 10)

    # Arrow down to feature engineering
    draw_arrow(12.0, 5.5, 12.0, 4.7)

    draw_box(9.0, 4.0, 4.5, 0.8, 'Feature Engineering\nFourier Terms / Holiday Mask', c_transform, 10, True)

    # Arrows from feature engineering to model families
    draw_arrow(9.0, 4.4, 7.5, 4.4)  # to label area

    # Row 3: Model families
    # Statistical
    draw_box(0.3, 2.5, 3.5, 1.2, 'Statistical Models\nAutoARIMA (SARIMAX)\nMSTL + ARIMA\nAutoTBATS', c_stat, 9, False)

    # ML / Tree
    draw_box(4.2, 2.5, 3.2, 1.2, 'Tree-Based Models\nLightGBM\nXGBoost\nCatBoost', c_ml, 9, False)

    # Neural
    draw_box(7.8, 2.5, 3.5, 1.2, 'Neural Models\nN-HiTS\nLSTM\nTFT (Transformer)', c_neural, 9, False)

    # Arrows from feature eng to model families
    draw_arrow(9.0, 4.0, 2.0, 3.7)
    draw_arrow(10.5, 4.0, 5.8, 3.7)
    draw_arrow(11.25, 4.0, 9.5, 3.7)

    # Row 4: Evaluation
    draw_box(3.0, 0.8, 5.5, 0.9, 'Rolling Cross-Validation (101 windows)\nMSE / MAPE + Confidence Intervals', c_eval, 10, True)

    draw_arrow(2.0, 2.5, 4.5, 1.7)
    draw_arrow(5.8, 2.5, 5.75, 1.7)
    draw_arrow(9.5, 2.5, 7.0, 1.7)

    # Execution annotations
    ax.text(2.0, 1.9, 'Parallel\n(CPU)', ha='center', va='center',
            fontsize=8, color=c_stat, fontstyle='italic')
    ax.text(5.8, 1.9, 'Parallel\n(CPU)', ha='center', va='center',
            fontsize=8, color=c_ml, fontstyle='italic')
    ax.text(9.5, 1.9, 'Sequential\n(GPU)', ha='center', va='center',
            fontsize=8, color=c_neural, fontstyle='italic')

    # Legend
    legend_items = [
        mpatches.Patch(color=c_data, label='Data Ingestion'),
        mpatches.Patch(color=c_transform, label='Preprocessing'),
        mpatches.Patch(color=c_stat, label='Statistical'),
        mpatches.Patch(color=c_ml, label='Tree-Based ML'),
        mpatches.Patch(color=c_neural, label='Neural'),
        mpatches.Patch(color=c_eval, label='Evaluation'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#cccccc', ncol=3)

    plt.tight_layout()
    plt.savefig('./docs/images/pipeline_architecture.png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Pipeline architecture diagram saved.")


def draw_model_comparison():
    """Draw a model comparison chart showing the 9 models across key dimensions."""
    models = [
        'AutoARIMA\n(SARIMAX)', 'MSTL\n+ ARIMA', 'AutoTBATS',
        'LightGBM', 'XGBoost', 'CatBoost',
        'N-HiTS', 'LSTM', 'TFT',
    ]

    categories = ['Interpretability', 'Nonlinear Capture', 'Training Speed', 'Seasonality Handling']

    # Approximate qualitative scores (1-5 scale)
    scores = np.array([
        # Interpret  Nonlinear  Speed  Seasonality
        [5, 1, 4, 3],   # SARIMAX
        [4, 2, 3, 5],   # MSTL+ARIMA
        [3, 2, 2, 5],   # TBATS
        [2, 4, 5, 3],   # LightGBM
        [2, 4, 4, 3],   # XGBoost
        [2, 4, 4, 3],   # CatBoost
        [1, 5, 2, 4],   # N-HiTS
        [1, 5, 1, 3],   # LSTM
        [1, 5, 1, 4],   # TFT
    ])

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(models))
    width = 0.18
    colors = ['#4A90D9', '#E67E22', '#2ECC71', '#9B59B6']

    for i, (cat, color) in enumerate(zip(categories, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, scores[:, i], width, label=cat, color=color, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Qualitative Score (1-5)', fontweight='bold')
    ax.set_title('Model Comparison Across Key Dimensions', fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 6)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1 (Low)', '2', '3', '4', '5 (High)'])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=10,
              framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Family separators
    ax.axvline(x=2.5, color='#cccccc', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=5.5, color='#cccccc', linestyle='--', linewidth=1, alpha=0.7)

    ax.text(1.0, 5.7, 'Statistical', ha='center', fontsize=10, color='#888888', fontstyle='italic')
    ax.text(4.0, 5.7, 'Tree-Based', ha='center', fontsize=10, color='#888888', fontstyle='italic')
    ax.text(7.0, 5.7, 'Neural', ha='center', fontsize=10, color='#888888', fontstyle='italic')

    plt.tight_layout()
    plt.savefig('./docs/images/model_comparison.png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Model comparison chart saved.")


def draw_cv_strategy():
    """Draw a diagram explaining the rolling cross-validation strategy."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    ax.text(6, 4.6, 'Rolling Cross-Validation Strategy (101 Windows)', ha='center',
            fontsize=14, fontweight='bold', color='#222222')

    colors_train = '#4A90D9'
    colors_test = '#E74C3C'
    gap_color = '#DDDDDD'

    n_folds = 5
    bar_height = 0.5
    gap = 0.15

    for i in range(n_folds):
        y = 3.8 - i * (bar_height + gap)
        train_width = 5 + i * 0.8
        test_start = train_width + 0.8
        test_width = 0.3

        # Train bar
        rect_train = mpatches.FancyBboxPatch(
            (0.8, y), train_width, bar_height,
            boxstyle='round,pad=0.05', facecolor=colors_train,
            edgecolor='white', linewidth=1, alpha=0.7 + i * 0.06,
        )
        ax.add_patch(rect_train)

        # Gap
        rect_gap = mpatches.FancyBboxPatch(
            (0.8 + train_width, y), 0.8, bar_height,
            boxstyle='round,pad=0.05', facecolor=gap_color,
            edgecolor='white', linewidth=1, alpha=0.5,
        )
        ax.add_patch(rect_gap)

        # Test bar
        rect_test = mpatches.FancyBboxPatch(
            (0.8 + test_start, y), test_width, bar_height,
            boxstyle='round,pad=0.05', facecolor=colors_test,
            edgecolor='white', linewidth=1, alpha=0.85,
        )
        ax.add_patch(rect_test)

        label = f'Fold {i + 1}' if i < 4 else f'Fold 101'
        ax.text(0.5, y + bar_height / 2, label, ha='right', va='center', fontsize=9, color='#555555')

    if n_folds == 5:
        # Ellipsis between fold 4 and fold 101
        y_dots = 3.8 - 3 * (bar_height + gap) - 0.15
        ax.text(5, y_dots, '. . .', ha='center', va='center', fontsize=14, color='#888888')

    # Legend
    legend_items = [
        mpatches.Patch(color=colors_train, label='Training Window (1680 obs)', alpha=0.85),
        mpatches.Patch(color=gap_color, label='Step Size (53 obs)', alpha=0.5),
        mpatches.Patch(color=colors_test, label='Forecast Horizon (h=1)', alpha=0.85),
    ]
    ax.legend(handles=legend_items, loc='lower center', fontsize=10,
              framealpha=0.9, edgecolor='#cccccc', ncol=3)

    # Annotations
    ax.annotate('refit=True\n(model retrained each fold)',
                xy=(9.5, 1.3), fontsize=9, color='#666666',
                ha='center', fontstyle='italic')

    plt.tight_layout()
    plt.savefig('./docs/images/cv_strategy.png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("CV strategy diagram saved.")


def draw_preprocessing_flow():
    """Draw the preprocessing transformations applied to the data."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    np.random.seed(42)

    # Panel 1: Raw signal with trend + seasonality
    t = np.linspace(0, 10, 500)
    trend = 0.3 * t
    seasonal_daily = 1.5 * np.sin(2 * np.pi * t / 1.0)
    seasonal_weekly = 0.8 * np.sin(2 * np.pi * t / 7.0)
    noise = np.random.normal(0, 0.3, len(t))
    raw = np.exp(trend + seasonal_daily + seasonal_weekly + noise + 3)

    # Add some outlier spikes
    outlier_idx = [50, 200, 350]
    raw_with_outliers = raw.copy()
    for idx in outlier_idx:
        raw_with_outliers[idx] *= 3.5

    axes[0].plot(t, raw_with_outliers, color='#4A90D9', linewidth=0.8, alpha=0.9)
    for idx in outlier_idx:
        axes[0].plot(t[idx], raw_with_outliers[idx], 'rv', markersize=8, alpha=0.8)
    axes[0].set_title('Raw Data', fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Order Count')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Panel 2: After log + outlier removal
    log_clean = np.log(raw)
    axes[1].plot(t, log_clean, color='#7B68EE', linewidth=0.8, alpha=0.9)
    axes[1].set_title('Log Transform + Hampel Filter', fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('log(y)')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Panel 3: Differenced (stationary)
    diff = np.diff(log_clean)
    axes[2].plot(t[1:], diff, color='#2ECC71', linewidth=0.8, alpha=0.9)
    axes[2].axhline(y=0, color='#cccccc', linestyle='--', linewidth=1)
    axes[2].set_title('Log-Differenced (Stationary)', fontweight='bold')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('diff(log(y))')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('./docs/images/preprocessing_flow.png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Preprocessing flow diagram saved.")


if __name__ == '__main__':
    draw_pipeline_diagram()
    draw_model_comparison()
    draw_cv_strategy()
    draw_preprocessing_flow()
    print("\nAll diagrams generated successfully.")
