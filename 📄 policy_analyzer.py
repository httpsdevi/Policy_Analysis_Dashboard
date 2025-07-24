# Policy Analysis Dashboard - Python Backend
# Author: Data Analyst
# Project: Advanced Insurance Analytics with ML-based Risk Prediction
# Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PolicyAnalyzer:
    """
    Advanced Policy Analysis System for Life Insurance Data
    Performs comprehensive analysis including lapse risk prediction,
    demographic segmentation, and premium optimization.
    """
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
    def load_and_prepare_data(self):
        """
        Simulate loading and cleaning real insurance policy data
        In production, this would connect to your data warehouse
        """
        print("📊 Loading policy data from data warehouse...")
        
        # Simulate realistic insurance policy dataset
        np.random.seed(42)
        n_policies = 10000
        
        # Generate synthetic but realistic policy data
        self.data = pd.DataFrame({
            'policy_id': [f'POL-2024-{1000+i}' for i in range(n_policies)],
            'age': np.random.normal(45, 12, n_policies).astype(int),
            'gender': np.random.choice(['M', 'F'], n_policies),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_policies),
            'policy_type': np.random.choice(['Term', 'Whole', 'Universal'], n_policies, p=[0.6, 0.25, 0.15]),
            'premium_amount': np.random.lognormal(10.5, 0.5, n_policies),
            'coverage_amount': np.random.lognormal(13.5, 0.7, n_policies),
            'years_active': np.random.gamma(2, 2, n_policies),
            'claims_count': np.random.poisson(0.3, n_policies),
            'payment_delays': np.random.poisson(0.8, n_policies),
            'customer_satisfaction': np.random.normal(7.5, 1.5, n_policies),
            'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Retired'], n_policies, p=[0.7, 0.2, 0.1]),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_policies, p=[0.3, 0.6, 0.1])
        })
        
        # Clean and prepare data
        self.data['age'] = np.clip(self.data['age'], 18, 80)
        self.data['customer_satisfaction'] = np.clip(self.data['customer_satisfaction'], 1, 10)
        self.data['claims_ratio'] = self.data['claims_count'] / np.maximum(self.data['years_active'], 1)
        
        # Create age groups for segmentation
        self.data['age_group'] = pd.cut(self.data['age'], 
                                       bins=[0, 25, 35, 45, 55, 65, 100], 
                                       labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'])
        
        # Create premium ranges
        self.data['premium_range'] = pd.cut(self.data['premium_amount'], 
                                           bins=[0, 25000, 50000, 100000, np.inf],
                                           labels=['<$25K', '$25K-$50K', '$50K-$100K', '$100K+'])
        
        print(f"✅ Loaded {len(self.data):,} policies with {self.data.shape[1]} features")
        return self.data.head()
    
    def calculate_lapse_risk(self):
        """
        Advanced lapse risk calculation using multiple factors
        """
        print("🎯 Calculating lapse risk scores...")
        
        # Risk factors with weights (based on actuarial research)
        risk_factors = {
            'age_risk': np.where(self.data['age'] < 30, 0.8, 
                        np.where(self.data['age'] > 60, 1.2, 1.0)),
            'payment_risk': np.minimum(self.data['payment_delays'] * 0.5, 2.0),
            'claims_risk': np.minimum(self.data['claims_ratio'] * 2.0, 1.5),
            'satisfaction_risk': (10 - self.data['customer_satisfaction']) * 0.2,
            'tenure_risk': np.where(self.data['years_active'] < 2, 1.5, 
                          np.where(self.data['years_active'] > 10, 0.5, 1.0))
        }
        
        # Calculate composite risk score
        base_risk = 5.0  # Base risk score
        total_risk = base_risk
        
        for factor, values in risk_factors.items():
            total_risk += values
        
        # Normalize to 0-10 scale
        self.data['lapse_risk_score'] = np.clip(total_risk, 0, 10)
        
        # Create risk categories
        self.data['risk_category'] = pd.cut(self.data['lapse_risk_score'],
                                           bins=[0, 4, 7, 10],
                                           labels=['Low', 'Medium', 'High'])
        
        print("✅ Risk scores calculated and categorized")
        
    def build_lapse_prediction_model(self):
        """
        Build ML model to predict policy lapse probability
        """
        print("🤖 Building ML lapse prediction model...")
        
        # Prepare features for ML model
        feature_columns = ['age', 'premium_amount', 'coverage_amount', 'years_active', 
                          'claims_count', 'payment_delays', 'customer_satisfaction']
        
        # Create target variable (simulated lapse based on risk score)
        self.data['will_lapse'] = (self.data['lapse_risk_score'] > 6.5) & (np.random.random(len(self.data)) < 0.3)
        
        # Encode categorical variables
        categorical_features = ['gender', 'region', 'policy_type', 'employment_status', 'marital_status']
        
        X = self.data[feature_columns].copy()
        
        for cat_col in categorical_features:
            le = LabelEncoder()
            X[f'{cat_col}_encoded'] = le.fit_transform(self.data[cat_col])
            self.encoders[cat_col] = le
        
        y = self.data['will_lapse'].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['lapse_model'] = scaler
        
        # Train Gradient Boosting model
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = gb_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Lapse Prediction Model Accuracy: {accuracy:.1%}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Lapse', 'Will Lapse']))
        
        # Store model and predictions
        self.models['lapse_predictor'] = gb_model
        self.data['lapse_probability'] = gb_model.predict_proba(scaler.transform(X))[:, 1]
        
        return accuracy
    
    def premium_analysis(self):
        """
        Comprehensive premium analysis by demographics
        """
        print("💰 Analyzing premium distributions...")
        
        # Premium statistics by demographics
        premium_stats = {
            'by_age_group': self.data.groupby('age_group')['premium_amount'].agg(['mean', 'median', 'count']),
            'by_region': self.data.groupby('region')['premium_amount'].agg(['mean', 'median', 'count']),
            'by_policy_type': self.data.groupby('policy_type')['premium_amount'].agg(['mean', 'median', 'count']),
            'by_risk_category': self.data.groupby('risk_category')['premium_amount'].agg(['mean', 'median', 'count'])
        }
        
        # Calculate retention rates
        retention_stats = self.data.groupby(['region', 'age_group']).agg({
            'will_lapse': lambda x: (1 - x.mean()) * 100,  # Retention rate %
            'premium_amount': 'mean',
            'lapse_risk_score': 'mean'
        }).round(2)
        
        retention_stats.columns = ['Retention_Rate_%', 'Avg_Premium', 'Avg_Risk_Score']
        
        print("✅ Premium analysis completed")
        return premium_stats, retention_stats
    
    def generate_insights(self):
        """
        Generate business insights for actuarial teams
        """
        print("💡 Generating business insights...")
        
        insights = []
        
        # High-value, low-risk segment analysis
        high_value_low_risk = self.data[
            (self.data['premium_amount'] > self.data['premium_amount'].quantile(0.8)) &
            (self.data['lapse_risk_score'] < 5)
        ]
        
        insights.append({
            'title': 'Premium Opportunity Segment',
            'finding': f"{len(high_value_low_risk):,} policies ({len(high_value_low_risk)/len(self.data):.1%}) represent high-value, low-risk segment",
            'avg_premium': f"${high_value_low_risk['premium_amount'].mean():,.0f}",
            'recommendation': 'Focus retention efforts and develop premium products for this segment'
        })
        
        # Regional performance analysis
        regional_performance = self.data.groupby('region').agg({
            'will_lapse': lambda x: (1 - x.mean()) * 100,
            'premium_amount': 'mean',
            'customer_satisfaction': 'mean'
        }).round(2)
        
        best_region = regional_performance['will_lapse'].idxmax()
        worst_region = regional_performance['will_lapse'].idxmin()
        
        insights.append({
            'title': 'Regional Performance Gap',
            'finding': f"{best_region} region outperforms {worst_region} by {regional_performance.loc[best_region, 'will_lapse'] - regional_performance.loc[worst_region, 'will_lapse']:.1f}% retention",
            'recommendation': f'Deploy {best_region} region best practices to {worst_region}'
        })
        
        # High-risk policy count
        high_risk_policies = self.data[self.data['lapse_risk_score'] > 7]
        immediate_action = self.data[self.data['lapse_probability'] > 0.7]
        
        insights.append({
            'title': 'Risk Intervention Opportunity',
            'finding': f"{len(high_risk_policies):,} policies need monitoring, {len(immediate_action):,} need immediate intervention",
            'potential_savings': f"${(immediate_action['premium_amount'].sum() * 0.7):,.0f} in annual premiums at risk",
            'recommendation': 'Launch targeted retention campaigns for high-probability lapse policies'
        })
        
        print("✅ Business insights generated")
        return insights
    
    def create_visualizations(self):
        """
        Create professional visualizations for the analysis
        """
        print("📈 Creating data visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Policy Analysis Dashboard - Key Insights', fontsize=16, fontweight='bold')
        
        # 1. Premium Distribution by Age Group
        age_premium = self.data.groupby('age_group')['premium_amount'].mean()
        axes[0, 0].bar(range(len(age_premium)), age_premium.values, color='steelblue', alpha=0.8)
        axes[0, 0].set_title('Average Premium by Age Group', fontweight='bold')
        axes[0, 0].set_xlabel('Age Group')
        axes[0, 0].set_ylabel('Average Premium ($)')
        axes[0, 0].set_xticks(range(len(age_premium)))
        axes[0, 0].set_xticklabels(age_premium.index, rotation=45)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Risk Distribution
        risk_counts = self.data['risk_category'].value_counts()
        colors = ['green', 'orange', 'red']
        axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
        axes[0, 1].set_title('Policy Risk Distribution', fontweight='bold')
        
        # 3. Regional Performance Heatmap
        regional_matrix = self.data.groupby(['region', 'age_group']).agg({
            'will_lapse': lambda x: (1 - x.mean()) * 100
        }).unstack(fill_value=0)
        
        sns.heatmap(regional_matrix.values, 
                   xticklabels=regional_matrix.columns.get_level_values(1),
                   yticklabels=regional_matrix.index,
                   annot=True, fmt='.1f', cmap='RdYlGn',
                   ax=axes[1, 0])
        axes[1, 0].set_title('Retention Rate % by Region & Age', fontweight='bold')
        axes[1, 0].set_xlabel('Age Group')
        axes[1, 0].set_ylabel('Region')
        
        # 4. Premium vs Risk Scatter
        axes[1, 1].scatter(self.data['lapse_risk_score'], self.data['premium_amount'], 
                          alpha=0.5, c=self.data['age'], cmap='viridis')
        axes[1, 1].set_title('Premium vs Risk Score (colored by age)', fontweight='bold')
        axes[1, 1].set_xlabel('Lapse Risk Score')
        axes[1, 1].set_ylabel('Premium Amount ($)')
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Add colorbar
        scatter = axes[1, 1].collections[0]
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Age', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig('policy_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved as 'policy_analysis_visualizations.png'")
        
        return fig
    
    def export_for_tableau(self):
        """
        Export cleaned data for Tableau dashboard creation
        """
        print("📊 Preparing data export for Tableau...")
        
        # Select key columns for Tableau
        tableau_data = self.data[[
            'policy_id', 'age', 'age_group', 'gender', 'region', 'policy_type', 
            'premium_amount', 'premium_range', 'coverage_amount', 'years_active',
            'claims_count', 'claims_ratio', 'payment_delays', 'customer_satisfaction',
            'employment_status', 'marital_status', 'lapse_risk_score', 'risk_category',
            'lapse_probability', 'will_lapse'
        ]].copy()
        
        # Add calculated fields that Tableau will use
        tableau_data['Premium_per_Coverage'] = tableau_data['premium_amount'] / tableau_data['coverage_amount']
        tableau_data['Customer_Lifetime_Value'] = tableau_data['premium_amount'] * tableau_data['years_active'] * (1 - tableau_data['lapse_probability'])
        tableau_data['Risk_Adjusted_Premium'] = tableau_data['premium_amount'] * (1 + tableau_data['lapse_risk_score'] / 10)
        
        # Export to CSV for Tableau
        tableau_data.to_csv('policy_data_for_tableau.csv', index=False)
        print("✅ Data exported to 'policy_data_for_tableau.csv' for Tableau dashboard")
        
        # Generate summary statistics for dashboard KPIs
        kpi_summary = {
            'Total_Policies': len(tableau_data),
            'Total_Premium_Volume': f"${tableau_data['premium_amount'].sum():,.0f}",
            'Average_Risk_Score': f"{tableau_data['lapse_risk_score'].mean():.1f}",
            'Retention_Rate': f"{(1 - tableau_data['will_lapse'].mean()) * 100:.1f}%",
            'High_Risk_Policies': len(tableau_data[tableau_data['risk_category'] == 'High']),
            'Average_Customer_Satisfaction': f"{tableau_data['customer_satisfaction'].mean():.1f}/10"
        }
        
        return tableau_data, kpi_summary
    
    def run_full_analysis(self):
        """
        Execute complete policy analysis workflow
        """
        print("🚀 Starting comprehensive policy analysis...\n")
        
        # Step 1: Load and prepare data
        data_sample = self.load_and_prepare_data()
        print(f"\n📋 Data Sample:\n{data_sample}")
        
        # Step 2: Calculate lapse risk
        self.calculate_lapse_risk()
        
        # Step 3: Build ML model
        model_accuracy = self.build_lapse_prediction_model()
        
        # Step 4: Premium analysis
        premium_stats, retention_stats = self.premium_analysis()
        
        # Step 5: Generate insights
        insights = self.generate_insights()
        
        # Step 6: Create visualizations
        self.create_visualizations()
        
        # Step 7: Export for Tableau
        tableau_data, kpi_summary = self.export_for_tableau()
        
        # Print summary report
        print("\n" + "="*80)
        print("📊 POLICY ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        print(f"\n📈 KEY PERFORMANCE INDICATORS:")
        for kpi, value in kpi_summary.items():
            print(f"  • {kpi.replace('_', ' ')}: {value}")
        
        print(f"\n🎯 BUSINESS INSIGHTS:")
        for i, insight in enumerate(insights, 1):
            print(f"\n  {i}. {insight['title']}")
            print(f"     Finding: {insight['finding']}")
            if 'recommendation' in insight:
                print(f"     Action: {insight['recommendation']}")
        
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"  • Lapse Prediction Accuracy: {model_accuracy:.1%}")
        print(f"  • Risk Segmentation: {len(self.data[self.data['risk_category'] == 'High'])} high-risk policies identified")
        
        print(f"\n📁 FILES GENERATED:")
        print(f"  • policy_analysis_visualizations.png (Charts for presentation)")
        print(f"  • policy_data_for_tableau.csv (Clean data for Tableau)")
        
        print("\n✅ Analysis complete! Ready for Tableau dashboard creation.")
        print("="*80)
        
        return {
            'data': self.data,
            'models': self.models,
            'insights': insights,
            'kpi_summary': kpi_summary,
            'tableau_data': tableau_data
        }

# Example usage and execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = PolicyAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_full_analysis()
    
    # Additional analysis examples
    print("\n🔍 ADDITIONAL ANALYSIS EXAMPLES:")
    
    # Customer segmentation analysis
    high_value_customers = results['data'][
        (results['data']['premium_amount'] > results['data']['premium_amount'].quantile(0.9)) &
        (results['data']['lapse_risk_score'] < 5)
    ]
    
    print(f"\n💎 Premium Customer Segment:")
    print(f"  • Count: {len(high_value_customers):,} policies")
    print(f"  • Average Premium: ${high_value_customers['premium_amount'].mean():,.0f}")
    print(f"  • Average Risk Score: {high_value_customers['lapse_risk_score'].mean():.1f}")
    print(f"  • Retention Rate: {(1 - high_value_customers['will_lapse'].mean()) * 100:.1f}%")
    
    # Regional opportunity analysis
    regional_opportunity = results['data'].groupby('region').agg({
        'premium_amount': ['count', 'mean', 'sum'],
        'lapse_risk_score': 'mean',
        'will_lapse': lambda x: (1 - x.mean()) * 100
    }).round(2)
    
    print(f"\n🌍 Regional Performance Summary:")
    print(regional_opportunity)
    
    print(f"\n🎉 Policy Analysis Dashboard - Python Backend Complete!")
    print(f"Ready to integrate with Tableau for advanced visualizations!")
