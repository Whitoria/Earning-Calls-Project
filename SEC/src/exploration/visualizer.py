import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = "SEC/data"
PLOT_DIR = "SEC/plots"
INPUT_FILE = f"{DATA_DIR}/nlp_metrics.parquet"

# Ensure plot subdirectories exist
os.makedirs(f"{PLOT_DIR}/time_series", exist_ok=True)
os.makedirs(f"{PLOT_DIR}/comparisons", exist_ok=True)
os.makedirs(f"{PLOT_DIR}/language_evolution", exist_ok=True)

def generate_plots():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        # Fallback for old pipeline runs
        fallback = f"{DATA_DIR}/nlp_metrics.csv"
        if os.path.exists(fallback):
             df = pd.read_csv(fallback)
        else:
            return
    else:
        df = pd.read_parquet(INPUT_FILE)
        
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df = df.sort_values('date')

    # Filter statistically noisy observations (< 50 words)
    df = df[df['word_count'] >= 50]
    
    print(f"Generating plots for {len(df)} significant sections...")

    # 1. Time Series: Risk Density over time (Avg per year)
    plt.figure(figsize=(12, 6))
    yearly_df = df.groupby(['year', 'ticker']).agg({'risk_density': 'mean', 'polarity_score': 'mean'}).reset_index()
    sns.lineplot(data=yearly_df, x='year', y='risk_density', hue='ticker', marker='o')
    plt.title("Yearly Average Risk Density by Company")
    plt.ylabel("Risk Words per 1000 Words")
    plt.xticks(yearly_df['year'].unique().astype(int))
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/time_series/risk_density_trend.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 1b. Time Series: Polarity over time
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=yearly_df, x='year', y='polarity_score', hue='ticker', marker='s')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Yearly Average Linguistic Polarity [-1 to 1] by Company")
    plt.ylabel("Normalized Polarity Score (Pos-Neg)/(Pos+Neg)")
    plt.xticks(yearly_df['year'].unique().astype(int))
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{PLOT_DIR}/time_series/polarity_trend.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Comparison: Readability Heatmap
    pivot_df = df.groupby(['ticker', 'section_name'])['readability_flesch'].mean().unstack()
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlOrRd", fmt=".1f", linewidths=.5)
    plt.title("Average Readability (Flesch Score) by Company & Section (Lower = Harder)")
    plt.savefig(f"{PLOT_DIR}/comparisons/readability_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Structural Difference: Document Length Distribution
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='ticker', y='word_count', hue='filing_type', split=True, inner="quart", palette="muted")
    plt.title("Filing Section Word Count Distribution by Ticker & Form")
    plt.yscale('log')
    plt.ylabel("Word Count (Log Scale)")
    plt.xticks(rotation=45)
    plt.savefig(f"{PLOT_DIR}/comparisons/word_count_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Polarity Ratio (Boxplot)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='ticker', y='polarity_score', palette="Set3", hue='ticker', legend=False)
    plt.axhline(0, color='red', linestyle='dashed')
    plt.title("Linguistic Polarity Score Distribution [-1 (Negative) to 1 (Positive)]")
    plt.xticks(rotation=45)
    plt.savefig(f"{PLOT_DIR}/comparisons/polarity_score_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Language Evolution: Top Keywords per Year Table
    KEYWORDS_FILE = f"{DATA_DIR}/keyword_year_table.parquet"
    if os.path.exists(KEYWORDS_FILE):
        keys_df = pd.read_parquet(KEYWORDS_FILE)
        
        # Improve text wrapping for keywords
        import textwrap
        keys_df['top_keywords'] = keys_df['top_keywords'].apply(lambda x: "\n".join(textwrap.wrap(x, width=60)))
        
        plt.figure(figsize=(14, len(keys_df) * 2 + 2))
        ax = plt.gca()
        ax.axis('off')
        ax.axis('tight')
        
        table = ax.table(cellText=keys_df.values, 
                         colLabels=keys_df.columns, 
                         loc='center', 
                         cellLoc='left',
                         colWidths=[0.1, 0.9])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 4)  # Increase vertical scale for spacing
        
        # Style headers
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4c72b0')
        
        plt.title("Linguistic Evolution: Top TF-IDF Keywords Per Year", pad=20, fontsize=16, weight='bold')
        plt.savefig(f"{PLOT_DIR}/language_evolution/keyword_evolution_table.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Financial Correlation: Polarity vs Net Income Growth
    CORR_FILE = "SEC/data/correlated_data.parquet"
    if os.path.exists(CORR_FILE):
        corr_df = pd.read_parquet(CORR_FILE)
        
        # Filter for recent meaningful growth data
        plot_df = corr_df.dropna(subset=['polarity_score', 'NetIncomeLoss_growth'])
        # Cap outliers for better visualization
        plot_df = plot_df[plot_df['NetIncomeLoss_growth'].between(-2, 2)]
        
        plt.figure(figsize=(12, 8))
        sns.regplot(data=plot_df, x='polarity_score', y='NetIncomeLoss_growth', 
                    scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title("Linguistic Sentiment vs. Realized Net Income Growth", fontsize=16, weight='bold')
        plt.xlabel("Polarity Score [-1 to 1]", fontsize=12)
        plt.ylabel("Net Income Growth (QoQ/YoY)", fontsize=12)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
        plt.savefig(f"{PLOT_DIR}/comparisons/sentiment_vs_performance_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Dual-Axis Time Series: NVDA Example
        nvda_df = corr_df[corr_df['ticker'] == 'NVDA'].sort_values('date')
        if not nvda_df.empty:
            fig, ax1 = plt.subplots(figsize=(14, 7))
            
            ax1.set_xlabel('Filing Date')
            ax1.set_ylabel('Polarity Score', color='tab:blue')
            ax1.plot(nvda_df['date'], nvda_df['polarity_score'], color='tab:blue', marker='o', linewidth=2, label='Polarity')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.axhline(0, color='blue', linestyle='--', alpha=0.2)
            
            ax2 = ax1.twinx()
            ax2.set_ylabel('Net Income ($B)', color='tab:green')
            ax2.bar(nvda_df['date'], nvda_df['NetIncomeLoss'] / 1e9, color='tab:green', alpha=0.3, width=20, label='Net Income')
            ax2.tick_params(axis='y', labelcolor='tab:green')
            
            plt.title("NVIDIA: Linguistic Sentiment vs. Actual Net Income", fontsize=16, weight='bold')
            fig.tight_layout()
            plt.savefig(f"{PLOT_DIR}/time_series/nvda_sentiment_profit_overlay.png", dpi=300, bbox_inches='tight')
            plt.close()

    print(f"Visualizations updated and saved to {PLOT_DIR} with higher DPI.")

if __name__ == "__main__":
    generate_plots()
