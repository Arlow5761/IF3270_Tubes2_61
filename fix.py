import tensorflow as tf
from tensorflow import keras

# 1. Import your FIXED custom layer
from src.compat.LocallyConnected2D import LocallyConnected2D

def fix_saved_model(broken_model_path, fixed_model_path):
    print(f"Loading broken model from: {broken_model_path}")
    
    # 2. Load the model. 
    # Because you used @keras.utils.register_keras_serializable(), 
    # just importing the fixed class is enough. Keras will use the new __init__
    # which safely converts the [3, 3] list into a (3, 3) tuple.
    model = keras.models.load_model(broken_model_path)
    
    # 3. Re-save the model. 
    # Now that it's correctly instantiated in memory, saving it will 
    # overwrite the file with a clean, working state.
    model.save(fixed_model_path)
    print(f"Fixed model saved to: {fixed_model_path}")

if __name__ == "__main__":
    # Example usage:
    # You can overwrite the original file by using the same path for both,
    # but it's safer to save to a new file first just in case.
    fix_saved_model(
        broken_model_path="models/lc_cnn_b3_flarge_k3_pavg.keras", 
        fixed_model_path="models/lc_cnn_b3_flarge_k3_pavg.keras"
    )