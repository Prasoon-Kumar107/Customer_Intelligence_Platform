import pickle as p
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# Model Saving
# ============================================================

def save_model(model, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    with open(file_path, "wb") as f:
        p.dump(model, f)


# ============================================================
# Model Loading
# ============================================================

def load_model(file_path):
    with open(file_path, "rb") as f:
        model = p.load(f)
    return model
