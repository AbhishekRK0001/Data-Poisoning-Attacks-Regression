# scripts/quick_test.py
import sys, os
# Add project root (parent of scripts/) to Python path so `import src.*` works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_utils import load_housing, get_split
from src.models import train_model, evaluate

# rest of your test script...


# scripts/quick_test.py
from src.data_utils import load_housing, get_split
from src.models import train_model, evaluate

X, y = load_housing()
print("Loaded X shape:", X.shape, "y shape:", y.shape)

X_train, X_test, y_train, y_test = get_split(X, y)
model = train_model("linear", X_train, y_train)
metrics = evaluate(model, X_test, y_test)
print("Baseline metrics:", metrics)
