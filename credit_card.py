# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

# %%
# Load the credit card dataset
print("Loading dataset...")
data = pd.read_csv('creditcard.csv')


# %%
# Display basic information about the dataset
print("\n--- Dataset Overview ---")
print(f"Dataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

# %%
print("\nLast 5 rows:")
print(data.tail())

# %%
data.info()

# %%
print("\nDataset information:")
print(data.info())


# %%
# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# %%
# Analyze class distribution
print("\n--- Class Distribution ---")
class_distribution = data['Class'].value_counts()
print(class_distribution)
print(f"Fraud cases: {class_distribution[1]} ({class_distribution[1]/len(data)*100:.2f}%)")
print(f"Normal cases: {class_distribution[0]} ({class_distribution[0]/len(data)*100:.2f}%)")

# %%
# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]  # Normal transactions
fraud = data[data.Class == 1]  # Fraudulent transactions
print(f"\nLegitimate transactions shape: {legit.shape}")
print(f"Fraudulent transactions shape: {fraud.shape}")


# %%
# Analyze transaction amounts
print("\n--- Transaction Amount Analysis ---")
print("Statistics for legitimate transactions:")
print(legit.Amount.describe())
print("\nStatistics for fraudulent transactions:")
print(fraud.Amount.describe())


# %%
# Compare means of features between classes
print("\n--- Feature Means by Class ---")
print(data.groupby('Class').mean())


# %%
# Create a balanced dataset using undersampling
# Note: This is a simple approach; consider using SMOTE or other techniques for better results
print("\n--- Creating Balanced Dataset ---")
print("Using undersampling for demonstration purposes...")
n_fraud = len(fraud)
legit_sample = legit.sample(n=n_fraud, random_state=42)
balanced_data = pd.concat([legit_sample, fraud], axis=0)
print(f"Balanced dataset shape: {balanced_data.shape}")
print(balanced_data['Class'].value_counts())

# %%
# Feature scaling - often important for logistic regression
print("\n--- Feature Scaling ---")
scaler = StandardScaler()
# Excluding 'Class' column and scaling all features
X = balanced_data.drop(columns='Class', axis=1)
Y = balanced_data['Class']
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# %%
# Split data into training and testing sets
print("\n--- Train-Test Split ---")
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled_df, Y, test_size=0.2, stratify=Y, random_state=42
)
print(f"Training set shape: {X_train.shape}, {Y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {Y_test.shape}")

# %%
# Train logistic regression model
print("\n--- Model Training ---")
# Increased max_iter to ensure convergence, added solver and C parameters
model = LogisticRegression(max_iter=10000, solver='liblinear', C=1.0, random_state=42)
model.fit(X_train, Y_train)

# %%
# Evaluate model performance on training data
print("\n--- Model Evaluation ---")
Y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, Y_train_pred)
print(f"Training accuracy: {training_accuracy:.4f}")

# %%
# Evaluate model performance on test data
Y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print(f"Test accuracy: {test_accuracy:.4f}")

# %%
# Detailed classification report
print("\nClassification Report:")
print(classification_report(Y_test, Y_test_pred))

# %%
# Generate confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(Y_test, Y_test_pred)
print(cm)


# %%
# Calculate precision, recall, f1-score
print("\nDetailed metrics:")
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# %%
# Save model for future use
print("\n--- Saving Model ---")
import joblib
joblib.dump(model, 'credit_card_fraud_model.pkl')
print("Model saved as 'credit_card_fraud_model.pkl'")

# %%
# Set the style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def visualize_credit_card_fraud(data, X, Y, X_train, Y_train, X_test, Y_test, model, Y_test_pred):
    """
    Generate comprehensive visualizations for credit card fraud detection analysis.
    
    Parameters:
    ----------
    data : pandas DataFrame
        Original dataset with features and Class column
    X : pandas DataFrame
        Features used for modeling
    Y : pandas Series
        Target variable (Class)
    X_train, Y_train : Training data
    X_test, Y_test : Test data
    model : trained model
    Y_test_pred : model predictions on test data
    """
    # Create a figure for multiple plots
    plt.figure(figsize=(20, 25))
    
    # 1. Class Distribution - Pie Chart
    plt.subplot(3, 2, 1)
    labels = ['Normal', 'Fraud']
    class_counts = data['Class'].value_counts()
    plt.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('Class Distribution in Original Dataset', fontsize=14)
    
    # 2. Transaction Amount Distribution - Fix for NaN/Inf values
    plt.subplot(3, 2, 2)
    # Check for and remove any infinite or NaN values
    clean_data = data.copy()
    clean_data['Amount'] = clean_data['Amount'].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Filter out any NaN values before plotting
    valid_data = clean_data.dropna(subset=['Amount'])
    
    # Use clipping to handle extreme values instead of log scale if necessary
    upper_limit = valid_data['Amount'].quantile(0.99)  # Use 99th percentile to avoid outliers
    valid_data['Amount_Clipped'] = valid_data['Amount'].clip(upper=upper_limit)
    
    # Plot with regular scale instead of log_scale
    sns.histplot(data=valid_data, x='Amount_Clipped', hue='Class', bins=50, kde=False)
    plt.title('Transaction Amount Distribution (Clipped at 99th percentile)', fontsize=14)
    plt.xlabel('Amount')
    plt.ylabel('Count')
    
    # 3. Feature Correlation Heatmap (using a subset of features for clarity)
    plt.subplot(3, 2, 3)
    try:
        # Select first 8 V features plus Amount and Class for clarity
        correlation_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'Amount', 'Class']
        correlation_features = [f for f in correlation_features if f in data.columns]
        correlation_data = data[correlation_features].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', cbar=True)
        plt.title('Feature Correlation Heatmap', fontsize=14)
    except Exception as e:
        plt.text(0.5, 0.5, f"Error in correlation plot: {str(e)}", horizontalalignment='center', fontsize=12)
    
    # 4. PCA Visualization of the dataset (2D projection)
    plt.subplot(3, 2, 4)
    try:
        # Apply PCA to reduce dimensions to 2 for visualization
        # Handle NaN values in X
        X_for_pca = X.fillna(X.mean())
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_for_pca)
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['Class'] = Y.values
        
        # Plot the 2D projection
        sns.scatterplot(x='PC1', y='PC2', hue='Class', data=pca_df, alpha=0.6, palette={0: '#66b3ff', 1: '#ff9999'})
        plt.title(f'PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})', fontsize=14)
    except Exception as e:
        plt.text(0.5, 0.5, f"Error in PCA plot: {str(e)}", horizontalalignment='center', fontsize=12)
    
    # 5. Confusion Matrix Heatmap
    plt.subplot(3, 2, 5)
    try:
        cm = confusion_matrix(Y_test, Y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    except Exception as e:
        plt.text(0.5, 0.5, f"Error in confusion matrix plot: {str(e)}", horizontalalignment='center', fontsize=12)
    
    # 6. ROC Curve
    plt.subplot(3, 2, 6)
    try:
        Y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, Y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic', fontsize=14)
        plt.legend(loc="lower right")
    except Exception as e:
        plt.text(0.5, 0.5, f"Error in ROC curve plot: {str(e)}", horizontalalignment='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('credit_card_fraud_visualization.png', dpi=300)
    plt.show()
    
    # 7. Feature Importance Plot (separate figure)
    plt.figure(figsize=(12, 8))
    try:
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Take top 15 or fewer if there aren't that many features
        top_n = min(15, len(feature_importance))
        sns.barplot(x='Importance', y='Feature', data=feature_importance[:top_n])
        plt.title(f'Top {top_n} Feature Importance', fontsize=14)
        plt.tight_layout()
        plt.savefig('credit_card_fraud_feature_importance.png', dpi=300)
        plt.show()
    except Exception as e:
        plt.text(0.5, 0.5, f"Error in feature importance plot: {str(e)}", horizontalalignment='center', fontsize=12)
        plt.tight_layout()
        plt.savefig('credit_card_fraud_feature_importance.png', dpi=300)
        plt.show()
    
    # 8. Precision-Recall Curve (separate figure)
    plt.figure(figsize=(10, 8))
    try:
        Y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(Y_test, Y_score)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig('credit_card_fraud_precision_recall.png', dpi=300)
        plt.show()
    except Exception as e:
        plt.text(0.5, 0.5, f"Error in precision-recall curve plot: {str(e)}", horizontalalignment='center', fontsize=12)
        plt.tight_layout()
        plt.savefig('credit_card_fraud_precision_recall.png', dpi=300)
        plt.show()
    
    # 9. Time Series of Transactions (separate figure)
    if 'Time' in data.columns:
        plt.figure(figsize=(15, 6))
        try:
            data_sorted = data.sort_values('Time')
            fraud_data = data_sorted[data_sorted['Class'] == 1]
            
            plt.plot(data_sorted['Time'], np.zeros(len(data_sorted)), 'b|', alpha=0.1, label='Normal')
            plt.plot(fraud_data['Time'], np.zeros(len(fraud_data)), 'r|', label='Fraud')
            plt.xlabel('Time (seconds from first transaction)')
            plt.title('Transaction Timeline', fontsize=14)
            plt.legend()
            plt.savefig('credit_card_fraud_timeline.png', dpi=300)
            plt.show()
        except Exception as e:
            plt.text(0.5, 0.5, f"Error in timeline plot: {str(e)}", horizontalalignment='center', fontsize=12)
            plt.tight_layout()
            plt.savefig('credit_card_fraud_timeline.png', dpi=300)
            plt.show()

# %%
visualize_credit_card_fraud(data, X, Y, X_train, Y_train, X_test, Y_test, model, Y_test_pred)