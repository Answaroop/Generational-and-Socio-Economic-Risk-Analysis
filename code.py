import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Dummy Data Creation
np.random.seed(42)
generations = ['Baby Boomers', 'Millennials', 'Gen Z']
socio_economic_classes = ['Rich', 'Poor']

# Create a realistic distribution for Risk_Taking_Score based on socio-economic classes
risk_taking_rich = np.random.normal(7, 1.5, 500)  # Mean 7, SD 1.5
risk_taking_poor = np.random.normal(5, 1.5, 500)  # Mean 5, SD 1.5
risk_taking_scores = np.concatenate([risk_taking_rich, risk_taking_poor])

# Generate socio-economic class and generation labels
generations_labels = np.random.choice(generations, 1000, p=[0.3, 0.5, 0.2])
socio_economic_labels = np.concatenate([
    np.full(500, 'Rich'),
    np.full(500, 'Poor')
])

# Generate initial income and investment preferences
initial_income = np.concatenate([
    np.random.normal(100000, 20000, 500),  # Rich: Mean income 100k, SD 20k
    np.random.normal(30000, 5000, 500)  # Poor: Mean income 30k, SD 5k
])
investment_preference = np.random.choice(['High Risk', 'Low Risk'], 1000, p=[0.6, 0.4])

# Combine into a DataFrame
data = {
    'Generation': generations_labels,
    'Socio_Economic_Class': socio_economic_labels,
    'Risk_Taking_Score': np.clip(risk_taking_scores, 1, 10),  # Ensure scores are between 1 and 10
    'Initial_Income': initial_income,
    'Investment_Preference': investment_preference
}

df = pd.DataFrame(data)

# Add calculated risk-taking preference based on socio-economic class
def calculate_risk_preference(row):
    if row['Socio_Economic_Class'] == 'Rich':
        return np.random.choice([1, 0], p=[0.8, 0.2])  # 80% calculated risk-takers
    else:
        return np.random.choice([1, 0], p=[0.55, 0.45])  # 55% calculated risk-takers

df['Calculated_Risk_Taker'] = df.apply(calculate_risk_preference, axis=1)

# Step 2: Data Analysis
# Group data by generation and socio-economic class
analysis = df.groupby(['Generation', 'Socio_Economic_Class'])['Calculated_Risk_Taker'].mean().reset_index()
analysis.rename(columns={'Calculated_Risk_Taker': 'Calculated_Risk_Taker_Percentage'}, inplace=True)

# Additional Statistical Analysis
# T-test for risk-taking scores between rich and poor
rich_scores = df[df['Socio_Economic_Class'] == 'Rich']['Risk_Taking_Score']
poor_scores = df[df['Socio_Economic_Class'] == 'Poor']['Risk_Taking_Score']
t_stat, p_value = ttest_ind(rich_scores, poor_scores)

# Chi-square test for independence between generation and calculated risk-taker
contingency_table = pd.crosstab(df['Generation'], df['Calculated_Risk_Taker'])
chi2, chi_p, dof, ex = chi2_contingency(contingency_table)

# Step 3: Regression Analysis
X = pd.get_dummies(df[['Generation', 'Socio_Economic_Class']], drop_first=True)
y = df['Risk_Taking_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Correlation between Risk_Taking_Score and Calculated_Risk_Taker
correlation, _ = pearsonr(df['Risk_Taking_Score'], df['Calculated_Risk_Taker'])

# Step 4: Income and Wealth Growth Projections
def simulate_wealth_growth(row):
    annual_return = 0.07 if row['Investment_Preference'] == 'High Risk' else 0.04
    years = 10
    return row['Initial_Income'] * ((1 + annual_return) ** years)

df['Projected_Wealth'] = df.apply(simulate_wealth_growth, axis=1)

wealth_analysis = df.groupby(['Generation', 'Socio_Economic_Class'])['Projected_Wealth'].mean().reset_index()

# Step 5: Risk-Adjusted Return Simulations
def simulate_risk_adjusted_return(row):
    base_return = 0.10 if row['Investment_Preference'] == 'High Risk' else 0.06
    risk_penalty = 0.02 * (10 - row['Risk_Taking_Score']) / 10
    return base_return - risk_penalty

df['Risk_Adjusted_Return'] = df.apply(simulate_risk_adjusted_return, axis=1)

return_analysis = df.groupby(['Generation', 'Socio_Economic_Class'])['Risk_Adjusted_Return'].mean().reset_index()

# Step 6: Visualizations
plt.figure(figsize=(10, 6))
sns.barplot(data=analysis, x='Generation', y='Calculated_Risk_Taker_Percentage', hue='Socio_Economic_Class', palette='viridis')
plt.title('Percentage of Calculated Risk Takers by Generation and Socio-Economic Class')
plt.ylabel('Calculated Risk Taker Percentage')
plt.xlabel('Generation')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Socio-Economic Class', fontsize=10)
plt.show()

# Distribution of Risk Taking Scores by Socio-Economic Class
plt.figure(figsize=(10, 6))
sns.kdeplot(rich_scores, label='Rich', shade=True, color='green')
sns.kdeplot(poor_scores, label='Poor', shade=True, color='blue')
plt.title('Distribution of Risk Taking Scores by Socio-Economic Class')
plt.xlabel('Risk Taking Score')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Wealth Growth Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=wealth_analysis, x='Generation', y='Projected_Wealth', hue='Socio_Economic_Class', palette='coolwarm')
plt.title('Projected Wealth Growth by Generation and Socio-Economic Class (10 years)')
plt.ylabel('Projected Wealth')
plt.xlabel('Generation')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Socio-Economic Class', fontsize=10)
plt.show()

# Risk-Adjusted Return Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=return_analysis, x='Generation', y='Risk_Adjusted_Return', hue='Socio_Economic_Class', palette='cubehelix')
plt.title('Risk-Adjusted Returns by Generation and Socio-Economic Class')
plt.ylabel('Risk-Adjusted Return')
plt.xlabel('Generation')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Socio-Economic Class', fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# Step 7: Comparative Analysis with Real Financial Benchmarks
# Simulating benchmark data (e.g., S&P 500 returns)
np.random.seed(43)
benchmark_returns = np.random.normal(0.08, 0.02, 1000)  # Mean 8%, SD 2%
df['Benchmark_Returns'] = np.random.choice(benchmark_returns, len(df))

# Calculate the difference between Risk-Adjusted Returns and Benchmark Returns
df['Return_Difference'] = df['Risk_Adjusted_Return'] - df['Benchmark_Returns']

# Grouped Analysis: Average Return Difference by Generation and Socio-Economic Class
benchmark_comparison = df.groupby(['Generation', 'Socio_Economic_Class'])['Return_Difference'].mean().reset_index()
benchmark_comparison.rename(columns={'Return_Difference': 'Avg_Return_Difference'}, inplace=True)

# Visualizing Return Difference
plt.figure(figsize=(10, 6))
sns.barplot(data=benchmark_comparison, x='Generation', y='Avg_Return_Difference', hue='Socio_Economic_Class', palette='mako')
plt.title('Average Return Difference from Benchmark by Generation and Socio-Economic Class')
plt.ylabel('Average Return Difference')
plt.xlabel('Generation')
plt.axhline(0, color='red', linestyle='--', linewidth=1, label='Benchmark Level')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Socio-Economic Class', fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# Correlation Analysis: Risk-Taking and Benchmark Comparison
correlation_benchmark, _ = pearsonr(df['Risk_Taking_Score'], df['Return_Difference'])

# Summary of Comparative Analysis
print("Benchmark Comparison Analysis:")
print(benchmark_comparison)
print("Correlation between Risk-Taking Scores and Return Difference from Benchmark:", correlation_benchmark)

# Additional Statistical Analysis: T-test for Return Differences
