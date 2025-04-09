import os
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
import xarray as xr
import rioxarray
from matplotlib.pyplot import plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

xr.set_options(keep_attrs=True, display_expand_data=True)


def initialize_random_forest_classifier(df, test_size=0.2, random_state=42):
    """
    Initialize a Random Forest classifier and split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset with features and labels.
        test_size float, optional (default=0.2): Proportion of the dataset to include in the test split.     
        random_state int, optional (default=42): Random state for reproducibility. 

    Returns:

    
    """
    # Split the data into features and labels
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    # Train the initial Random Forest classifier
        # Train the Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )

    rf_classifier.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return rf_classifier, X_train, X_test, y_train, y_test, accuracy


def get_feature_importance(model, df):
    """
    Get feature importance from the trained Random Forest model.
    
    Args:
        model (RandomForestClassifier): Trained Random Forest classifier.
        df (pd.DataFrame): DataFrame containing the training dataset with features and labels.

    Returns:
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': df.columns[:-1],
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
   
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))   
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title(f'Feature Importance Plot')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()

    return fig, feature_importance