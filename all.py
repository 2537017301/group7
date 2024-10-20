import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, anderson, ttest_ind, mannwhitneyu, linregress, chi2_contingency
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class DataInspection:
    def __init__(self):
        self.df = None

    def load_csv(self, file_path):
        """Load CSV file into a DataFrame"""
        try:
            self.df = pd.read_csv(file_path)
            if self.df.empty:
                raise ValueError("The CSV file is empty.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File not found at {file_path}")
        except pd.errors.ParserError:
            raise pd.errors.ParserError(f"Error: Could not parse the CSV file at {file_path}")

    def handle_missing_values(self, col):
        """Handle missing values in a column based on percentage"""
        missing_percentage = self.df[col].isna().sum() / len(self.df) * 100
        if missing_percentage > 50:
            self.df.drop(columns=[col], inplace=True)
            return False
        else:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            return True

    def numeric_columns(self):
        """Return a list of numeric columns in the DataFrame"""
        return [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]

    def plot_histogram(self, col):
        """Plot a histogram for the given column"""
        plt.hist(self.df[col].dropna())
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    def plot_boxplot(self, x_col, y_col):
        """Plot a box plot for the given columns"""
        self.df.boxplot(column=[y_col], by=x_col)
        plt.title(f"Boxplot of {y_col} by {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.suptitle('')
        plt.show()

    def plot_bar_chart(self, col):
        """Plot a bar chart for the given column"""
        value_counts = self.df[col].value_counts()
        value_counts.plot(kind='bar')
        plt.title(f'Bar chart of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
class DataAnalysis:
    def __init__(self):
        self.df = None
        self.column_types = {}
    def dataset_loading(self, file_path):
        """Load the dataset from a CSV file."""
        self.df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")

    def list_column_types(self):
        """Check column types and categorize them."""
        self.column_types = {
            'interval': [],
            'numeric_ordinal': [],
            'non_numeric_ordinal': [],
            'nominal': []
        }
        
        for col in self.df.columns:
            unique_values = self.df[col].nunique()
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if unique_values > 10:
                    self.column_types['interval'].append(col)
                else:
                    self.column_types['numeric_ordinal'].append(col)
            else:
                if unique_values <= 10:
                    self.column_types['nominal'].append(col)
                else:
                    self.column_types['non_numeric_ordinal'].append(col)

        return self.column_types

    def select_variable(self, data_type, max_categories=None, allow_skip=False):
        available_columns = [col for col in self.column_types[data_type]
                             if max_categories is None or self.df[col].nunique() <= max_categories]

        if not available_columns:
            if allow_skip:
                return None
            raise ValueError(f"No available columns of type {data_type} with max categories {max_categories}.")

        print(f"Available columns of type {data_type}: {available_columns}")
        selected_var = input("Select a variable from the list: ")
        
        while selected_var not in available_columns:
            selected_var = input(f"Invalid selection. Choose from {available_columns}: ")

        return selected_var

    def check_normality(self, column, size_limit=2000):
        if len(self.df[column]) > size_limit:
            stat, _ = anderson(self.df[column].dropna())
            print(f"Anderson-Darling Test: statistic = {stat}")
            return 'Anderson-Darling', stat
        else:
            stat, p_value = shapiro(self.df[column].dropna())
            print(f"Shapiro-Wilk Test: statistic = {stat:.4f}, p-value = {p_value:.15f}")
            return 'Shapiro-Wilk', stat, p_value

    def perform_regression(self, x_var, y_var):
        X = self.df[x_var].dropna()
        Y = self.df[y_var].dropna()
        
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]

        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
        print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}, R-squared: {r_value**2:.4f}, P-value: {p_value:.15f}")

    def t_test_or_mannwhitney(self, continuous_var, categorical_var):
        groups = [group[continuous_var].dropna() for name, group in self.df.groupby(categorical_var)]
        normality_test = self.check_normality(self.df[continuous_var])

        if normality_test[0] == 'Shapiro-Wilk' and normality_test[2] > 0.05:
            stat, p_value = ttest_ind(*groups)
            print(f"t-Test: Statistic = {stat:.4f}, p-value = {p_value:.15f}")
        else:
            stat, p_value = mannwhitneyu(*groups)
            print(f"Mann-Whitney U Test: Statistic = {stat:.4f}, p-value = {p_value:.15f}")

    def chi_square_test(self, categorical_var_1, categorical_var_2):
        contingency_table = pd.crosstab(self.df[categorical_var_1], self.df[categorical_var_2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"chi2 = {chi2:.4f}, p-value = {p:.15f}, degrees of freedom = {dof}")

class SentimentAnalysis: 
    def __init__(self):
        self.df = None

    def load_data(self, path):
        """Load the dataset from the specified path."""
        self.df = pd.read_csv(path)
        print(f"Dataset loaded successfully from {path}")

    def get_text_columns(self):
        """Select text columns and calculate average entry length and unique entries."""
        text_columns = []
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                avg_length = self.df[column].apply(lambda x: len(str(x))).mean()
                unique_entries = self.df[column].nunique()
                text_columns.append([column, avg_length, unique_entries])
        
        text_df = pd.DataFrame(text_columns, columns=['Column Name', 'Average Entry Length', 'Unique Entries'])
        return text_df

    def vader_sentiment_analysis(self, data):
        """Perform VADER sentiment analysis."""
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        sentiments = []

        for entry in data:
            score = analyzer.polarity_scores(entry)['compound']
            scores.append(score)
            if score >= 0.05:
                sentiments.append('positive')
            elif score <= -0.05:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')

        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        """Perform TextBlob sentiment analysis."""
        scores = []
        sentiments = []
        subjectivity = []

        for entry in data:
            analysis = TextBlob(entry)
            score = analysis.sentiment.polarity
            subject = analysis.sentiment.subjectivity
            scores.append(score)
            subjectivity.append(subject)
            
            if score > 0:
                sentiments.append('positive')
            elif score == 0:
                sentiments.append('neutral')
            else:
                sentiments.append('negative')

        return scores, sentiments, subjectivity

    def distilbert_sentiment_analysis(self, data):
        """Perform DistilBERT sentiment analysis using transformers pipeline."""
        if pipeline is None:
            raise ImportError("Transformers is not installed")

        distilbert_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        scores = []
        sentiments = []

        for entry in data:
            result = distilbert_pipeline(entry)[0]
            label = result['label']
            score = result['score']
            scores.append(score)

            if '4' in label or '5' in label:
                sentiments.append('positive')
            elif '3' in label:
                sentiments.append('neutral')
            else:
                sentiments.append('negative')

        return scores, sentiments

def main():
    """Main function to handle user input and execute analysis."""
    # Data Inspection
    inspection = DataInspection()
    file_path = input("Enter the path to your dataset: ")
    inspection.load_csv(file_path)

    # Data Analysis
    analysis = DataAnalysis()
    analysis.dataset_loading(file_path)
    analysis.list_column_types()

    # Sentiment Analysis
    sa = SentimentAnalysis()
    path = input("Enter the path to your sentiment dataset: ")
    sa.load_data(path)
    
    text_columns_df = sa.get_text_columns()
    print("\nText columns available for sentiment analysis:")
    print(text_columns_df)
    
    column_name = input("\nEnter the column name to analyze: ")
    print("\nChoose the type of sentiment analysis:")
    print("1. VADER")
    print("2. TextBlob")
    print("3. DistilBERT")
    choice = input("Enter your choice (1/2/3): ")
    
    if column_name in sa.df.columns:
        data = sa.df[column_name].dropna()
        if choice == '1':
            print("\nPerforming VADER sentiment analysis...")
            scores, sentiments = sa.vader_sentiment_analysis(data)
            result_df = pd.DataFrame({'Text': data, 'VADER Score': scores, 'Sentiment': sentiments})
        elif choice == '2':
            print("\nPerforming TextBlob sentiment analysis...")
            scores, sentiments, subjectivity = sa.textblob_sentiment_analysis(data)
            result_df = pd.DataFrame({'Text': data, 'Polarity Score': scores, 'Sentiment': sentiments, 'Subjectivity': subjectivity})
        elif choice == '3':
            print("\nPerforming DistilBERT sentiment analysis...")
            scores, sentiments = sa.distilbert_sentiment_analysis(data)
            result_df = pd.DataFrame({'Text': data, 'DistilBERT Score': scores, 'Sentiment': sentiments})
        else:
            print("Invalid choice. Exiting.")
            return
        
        print("\nSentiment analysis result:")
        print(result_df)
    else:
        print(f"Column {column_name} not found in the dataset.")

if __name__ == '__main__':
    main()
