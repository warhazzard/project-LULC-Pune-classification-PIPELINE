import os
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

xr.set_options(keep_attrs=True, display_expand_data=True)


def initialize_random_forest_classifier(df, test_size=0.2, random_state=42):
    """
    Initialize a Random Forest classifier and split the dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): DataFrame containing the dataset with features and labels.
        test_size float, optional (default=0.2): Proportion of the dataset to include in the test split.     
        random_state int, optional (default=42): Random state for reproducibility. 

    Returns:
        rf_classifier (RandomForestClassifier): Trained Random Forest classifier.
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        accuracy (float): Accuracy of the model on the test set.
    
    """
    X = df.drop(columns=['label'])
    y = df['label']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, stratify=y, random_state=random_state)

    # Train the initial Random Forest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1
    )

    rf_classifier.fit(X_train, y_train)

    # Evaluate the model - for initial testing
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return rf_classifier, X_train, X_test, y_train, y_test, accuracy, le.classes_


def get_feature_importance(model, df, output_path=None):
    """
    Get feature importance from the trained Random Forest model.
    
    Args:
        model (RandomForestClassifier): Trained Random Forest classifier.
        df (pd.DataFrame): DataFrame containing the training dataset with features and labels.

    Returns:
        fig (plt.Figure): Figure object containing the feature importance plot.
        feature_importance (pd.DataFrame): DataFrame containing feature importances sorted by importance.
    """
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance = pd.DataFrame({
        'Feature': df.drop(columns='label').columns,
        'Importance': importances
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
   
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))   
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title(f'Feature Importance Plot')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        feature_importance.to_csv(f'{output_path}feature_importance.csv', index=False)
        fig.savefig(f'{output_path}feature_importance_plot.png', dpi=300)
    
    return fig, feature_importance


def get_max_safe_folds(df, min_samples_per_fold=100):
    """
    Get the maximum number of safe folds for cross-validation based on the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with features and labels.

    Returns:
        max_safe_folds (int): Maximum number of safe folds for cross-validation.
    """

    class_counts = df['label'].value_counts()
    rarest_class_count = class_counts.min()
    max_folds = rarest_class_count // min_samples_per_fold
    
    if max_folds < 3:
        print(f"Class counts: {class_counts}\n")
        raise ValueError(f"Not enough samples in the rarest class to create {min_samples_per_fold} samples per fold.")
    else:
        print(f"Class counts: {class_counts}")
        print(f"\nMax recommended folds: {max_folds}")
        max_safe_folds = max_folds
    
    return max_safe_folds, class_counts


def tune_hyperparameters(X, y, max_safe_folds):
    """
    Tune hyperparameters for the Random Forest classifier using GridSearchCV.
    
    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training labels.

    Returns:
        best_model (RandomForestClassifier): Best Random Forest classifier after hyperparameter tuning.
        best_params (dict): Best hyperparameters found during tuning.
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
    k_fold_cv = StratifiedKFold(n_splits=max_safe_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=k_fold_cv, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best Hyperparameters:")
    print(best_params)

    return best_model, best_params, grid_search.cv_results_


def evaluate_best_model(best_model, X_test, y_test, plot_confusion_matrix=True):
    """
    Evaluate the trained model using various metrics.

    Args:
        best_model (RandomForestClassifier): Trained Random Forest classifier.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
    
    Returns:
        metrics (dict): Dictionary of evaluation metrics:
            accuracy (float): Accuracy of the model on the test set.
            confusion_mat (np.ndarray): Confusion matrix.
            classification_rep (str): Classification report as a string.

    """
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\n Classification Report: \n")
    print(classification_rep)

    if plot_confusion_matrix:
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()
        plt.show()
        

    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'classification_report': classification_rep
    }

    return metrics

def save_trained_model(model, output_path):
    """
    Save the trained model to a file.

    Args:
        model (Classifier): Trained Random Forest classifier - '.pkl'.
        output_path (str): Path to save the model.
    
    Returns: 
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")
    return None


def load_trained_model(model_path):
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        model (Classifier): Loaded Random Forest classifier '.pkl'.
    """
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
def classify_raster(model, raster_ds, output_path=None):
    """
    Classify a raster using the trained model.

    Args:
        model (RandomForestClassifier): Trained Random Forest classifier.
        raster_ds (rioxarray.DataArray): Input multi-band raster data.
        output_path (str): Path to save the classified raster.

    Returns:
        None (if output_path is given) or classified_da (rioxarray.DataArray): Classified raster as a DataArray.
    """
    bands = raster_ds['band'].values

    bands_array = []
    for band in bands:
        bands_array.append(raster_ds.sel(band=band).values)

    band_stack = np.stack(bands_array, axis=0)
    height, width = band_stack.shape[1], band_stack.shape[2]
    flatten_bands = band_stack.reshape(band_stack.shape[0], -1).T  

    # Create a mask for valid pixels - filtering out values other specified in condition and avoid breaking up the classifier results
    valid_mask = np.all(flatten_bands > 0, axis=1)  

    # Initialize the classification result array and proceed to classify
    # classification = np.zeros(height * width, dtype=np.uint8)
    # classification[valid_mask] = model.predict(flatten_bands[valid_mask])
    # classified_image = classification.reshape(height, width)

    # Predict with the model
    predicted_codes = model.predict(flatten_bands) 
    classification = np.zeros(height * width, dtype=np.uint8)
    classification[valid_mask] = predicted_codes[valid_mask]
    classified_image = classification.reshape(height, width)

    classified_da = rioxarray.DataArray(classified_image, dims=["y", "x"], coords={"y": raster_ds.y, "x": raster_ds.x})
    classified_da.rio.write_crs(raster_ds.rio.crs, inplace=True)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        classified_da.rio.to_raster(output_path)
        print(f"Classified raster saved to {output_path}")
        return None
    
    return classified_da
    

# tree_depths = [estimator.tree_.max_depth for estimator in rf_classifier.estimators_]
# print("Max depth per tree (sample):", tree_depths[:10])  
# print("Average tree depth:", np.mean(tree_depths))
# print("Max tree depth:", np.max(tree_depths))
# print("Min tree depth:", np.min(tree_depths))