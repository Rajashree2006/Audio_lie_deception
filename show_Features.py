# show_features.py
import joblib
import os

print("Current directory:", os.getcwd())

try:
    pipeline = joblib.load("rf_lie_model.pkl")
    print("\nPipeline loaded OK")

    # Try modern way (scikit-learn >= 1.0)
    if hasattr(pipeline.named_steps['model'], 'feature_names_in_'):
        features = pipeline.named_steps['model'].feature_names_in_
        print("\nFeature names (from feature_names_in_):")
    else:
        # Fallback: get from the pipeline's input features (older style or when not set)
        print("\nNo feature_names_in_ → using columns from last fit (approximate)")
        # If you still have X_train in scope — but since it's a saved model, we try another way
        try:
            # In many cases the imputer or scaler has it
            features = pipeline.named_steps['scaler'].get_feature_names_out()
        except:
            features = None

    if features is not None:
        print("\n".join(features))
        print(f"\nTotal features: {len(features)}")
    else:
        print("Could not recover feature names automatically.")

    # Bonus: show classes
    print("\nClasses:", pipeline.named_steps['model'].classes_)

except Exception as e:
    print("Error:", str(e))