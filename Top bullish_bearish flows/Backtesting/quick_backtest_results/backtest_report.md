# Options Flow Scoring Backtest Report

**Generated:** 2026-01-13 21:13:38

## Dataset Summary

- **Total data points:** 970
- **Unique tickers:** 227
- **Date range:** 2024-04-30 to 2025-12-30
- **Holding periods tested:** [3, 5, 7, 10, 14, 21]

## Correlation Analysis (Quant Methodology)

### Swing Trading Periods (3-7 days)

| holding_period | ic | ic_tstat | hit_rate | pearson_correlation | sample_size |
| --- | --- | --- | --- | --- | --- |
| 3.0 | 0.0062 | 0.1749 | 50.7216 | 0.0181 | 970.0 |
| 5.0 | 0.0585 | 1.7036 | 51.6495 | 0.0317 | 970.0 |
| 7.0 | 0.0813 | 2.6751 | 50.7216 | 0.0332 | 970.0 |

- **Average Information Coefficient (IC):** 0.0487
- **Average IC t-statistic:** 1.52
- **Average Hit Rate:** 51.0%
- **Average Pearson Correlation:** 0.0277

**WEAK predictive signal** - Positive IC but low significance
   Positive flow scores weakly predict positive returns

**Marginal hit rate** (51.0%) - Slightly better than random

### All Periods

| holding_period | pearson_correlation | pearson_pvalue | spearman_correlation | spearman_pvalue | ic | ic_tstat | hit_rate | sample_size | num_dates | ic_positive_days | ic_negative_days | ic_positive_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | -0.0129 | 0.6884 | -0.0084 | 0.7941 | 0.0545 | 1.7013 | 46.8041 | 970.0 | 27.0 | 18.0 | 9.0 | 66.6667 |
| 2.0 | -0.0031 | 0.9237 | -0.0121 | 0.7074 | -0.0182 | -0.5005 | 49.3814 | 970.0 | 27.0 | 12.0 | 15.0 | 44.4444 |
| 3.0 | 0.0181 | 0.5732 | 0.0072 | 0.8217 | 0.0062 | 0.1749 | 50.7216 | 970.0 | 27.0 | 16.0 | 11.0 | 59.2593 |
| 4.0 | 0.0324 | 0.3137 | 0.0148 | 0.646 | 0.038 | 1.0448 | 51.6495 | 970.0 | 27.0 | 16.0 | 11.0 | 59.2593 |
| 5.0 | 0.0317 | 0.3233 | 0.0209 | 0.5154 | 0.0585 | 1.7036 | 51.6495 | 970.0 | 27.0 | 18.0 | 9.0 | 66.6667 |
| 6.0 | 0.029 | 0.3663 | -0.0085 | 0.7919 | 0.0762 | 2.654 | 49.5876 | 970.0 | 27.0 | 22.0 | 5.0 | 81.4815 |
| 7.0 | 0.0332 | 0.3015 | 0.0037 | 0.9094 | 0.0813 | 2.6751 | 50.7216 | 970.0 | 27.0 | 19.0 | 8.0 | 70.3704 |
| 8.0 | 0.0189 | 0.5558 | -0.011 | 0.7318 | 0.05 | 1.4023 | 50.9278 | 970.0 | 27.0 | 16.0 | 11.0 | 59.2593 |
| 9.0 | 0.0083 | 0.7959 | -0.0098 | 0.7597 | 0.0412 | 1.1884 | 50.5155 | 970.0 | 27.0 | 16.0 | 11.0 | 59.2593 |
| 10.0 | 0.0134 | 0.6951 | -0.0051 | 0.8826 | 0.0333 | 0.8084 | 50.2917 | 857.0 | 24.0 | 15.0 | 9.0 | 62.5 |

## Methodology

This backtest tests the **directional relationship** between flow scores and returns:
- **Positive flow scores** (bullish flow) should predict **positive returns**
- **Negative flow scores** (bearish flow) should predict **negative returns**

The correlation coefficient measures this directional relationship:
- **Positive correlation** = bullish flow predicts positive returns (desired)
- **Negative correlation** = bullish flow predicts negative returns (inverse/contrarian signal)

Flow scores preserve sign: positive = bullish flow, negative = bearish flow.
Unlike ranking-based approaches, this tests if the magnitude and direction
of flow scores predict the magnitude and direction of returns.

## Optimal Parameters

### Best Parameters Found

| Parameter | Value |
| --- | --- |
| `BID_WEIGHT` | 0.5006009949771377 |
| `DTE_LAMBDA` | 0.06983643719274202 |
| `SECONDARY_CLASS_WEIGHT` | 0.2827394726652016 |
| `MIN_SECONDARY_THRESHOLD` | 0.24681264951009116 |
| `PRICE_CAP_ADJUSTMENT` | True |

### Optimization Statistics

- **Best score:** 12.0869
- **Average score:** 9.9021
- **Score improvement:** +22.06% over average

## AI Interpretation (DeepSeek - Cloud API)

Okay, let's break down these options flow backtest results. Here's a data-driven analysis focusing on your requested points:

**1. Predictive Signal Assessment: Weak Signal**

*   **Information Coefficient (IC):** The IC of 0.0487 is extremely low. This is the most critical metric for evaluating a predictive model's skill. An IC this low indicates that the model's predictions are only slightly better than random chance (which would be an IC near 0). A commonly cited threshold considers an IC > 0.05 as "good," but even at this higher threshold, 0.0487 is marginal at best. It suggests the model is barely capturing any exploitable edge in the market.
*   **T-statistic:** The t-stat of 1.52 provides some statistical validation. It indicates that the IC is significantly different from zero at the 1.5% two-tailed significance level (roughly). However, statistical significance does not equate to practical significance or economic value, especially with such a low IC magnitude. The signal is weak and likely noisy.
*   **Hit Rate:** A 51% hit rate is only marginally better than random chance (50%). This further confirms the lack of a strong predictive edge. While positive, it doesn't justify the potential costs of trading based on this signal.
*   **Correlation:** The correlation of 0.0277 between the model's predictions and the actual outcome is very low. This means the model's directional predictions (buy/hold/sell) align with market movements very infrequently, reinforcing the weak predictive power.

**Overall Assessment:** The predictive signal generated by this model is assessed as **Weak**. While statistically distinguishable from pure randomness, the practical value is minimal due to the near-zero IC and low hit rate.

**2. Parameter Insights:**

The parameters provided (`BID_WEIGHT`, `DTE_LAMBDA`, `SECONDARY_CLASS_WEIGHT`, `MIN_SECONDARY_THRESHOLD`, `PRICE_CAP_ADJUSTMENT`) are the hyperparameters optimized during the model development. The optimization process found a specific combination yielding the best backtest score (likely a profit metric like Sharpe Ratio or total return).

*   **BID_WEIGHT (0.5006):** This parameter adjusts the relative importance of bid-side flow versus ask-side flow. A value of 0.5 is effectively neutral, suggesting the model doesn't strongly favor one side over the other based on the bid weight parameter itself. The slight deviation from 0.5 might be insignificant given the weak overall signal.
*   **DTE_LAMBDA (0.0698):** This likely controls the weighting of options with different time-to-expiry (DTE). A lower lambda means more weight is given to near-term options, while a higher lambda means more weight on longer-term options. The value 0.0698 suggests a slight preference for longer-term options, but its impact might be overshadowed by the weak signal.
*   **Secondary Class Weight (0.2827) & Threshold (0.2468):** These parameters likely relate to handling complex option contracts (e.g., LEAPs, different strike classes). The weight (0.2827) might be the influence given to secondary class signals, and the threshold (0.2468) might be a minimum signal strength required to consider a secondary class. Their specific impact needs context, but they are part of the model's structure.
*   **PRICE_CAP_ADJUSTMENT (true):** This indicates the model incorporates an adjustment for price effects, likely to mitigate distortions from high-priced stocks.

**Interpretation:** While the specific values are optimized, the overall predictive power (IC) remains very low. This suggests that the *combination* of these parameters, while theoretically sound for potentially capturing flow information, does not generate a strong predictive edge in this specific application (ICM-like signals).

**3. Overfitting Concerns:**

*   **Optimization Statistics:** The stark contrast between the best score (12.0869) and the average score (9.9021) is a major red flag for overfitting. The model likely found a combination of parameters that performs exceptionally well on the *entire dataset* used for optimization, but this performance does not generalize well to unseen data (represented by the average score over multiple optimization runs or different datasets). The standard deviation (1.9805) is relatively low, suggesting the optimization landscape might be somewhat smooth, but the gap between best and average is still concerning.
*   **Sample Size:** While 970 data points provide some statistical power, the optimization process itself consumes data. The risk remains that the model has essentially memorized patterns specific to the optimization period rather than learning generalizable features.
*   **Actionable Concern:** This model likely suffers from significant overfitting. The high best score might be misleading. Cross-validation during optimization (e.g., nested cross-validation) is crucial but wasn't mentioned here. Relying solely on the "best" score without validation is dangerous.

**4. Trading Recommendations:**

Based on the weak predictive signal and overfitting concerns:

*   **Do Not Trade Based on This Model:** The evidence strongly suggests that this model does not provide a reliable edge for trading decisions. The low IC and hit rate indicate that any trading strategy based on it would likely result in near-random performance, or worse, due to transaction costs and potential overfitting penalties.
*   **Focus on Feature Engineering:** If the goal is to build an ICM-like strategy, the weak signal implies the *features* themselves (perhaps the way flow is measured or aggregated) might need significant improvement. Consider different ways to capture flow, incorporate more diverse market context, or combine flow data with other fundamental/technical indicators.
*   **Re-evaluate Optimization:** Address the overfitting concern. Implement rigorous cross-validation during the optimization process. Use separate validation sets to estimate out-of-sample performance. Consider regularization techniques if applicable to the model type.
*   **Explore Alternative Models:** Investigate different machine learning models or statistical approaches that might capture the underlying market dynamics better. Perhaps simpler models could sometimes perform better if the data doesn't inherently contain strong predictive signals.
*   **Consider Costs:** Even a slight edge (IC ~0.05) can justify trading if costs are low and capitalization is high. However, with an IC this low, the risk of losses due to random chance and transaction costs is very high. Prudent risk management would strongly caution against deployment.

**Conclusion:** This backtest results indicate a model with a **Weak** predictive signal (low IC, low hit rate, low correlation). While the parameters are optimized, the overall performance suggests potential **overfitting**. **Trading recommendations are strongly discouraged** based on this model alone. Focus should be shifted towards improving the underlying features or exploring fundamentally different modeling approaches.

## Recommendations

1. **Best holding period:** 7 days (correlation: 0.033)
2. **Use optimized parameters** for improved correlation

---
