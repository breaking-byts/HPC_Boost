import time
import numpy as np
from xgboost import XGBClassifier

def test_speed():
    # 2700 samples, 24 features (4 events * 6 stats)
    X = np.random.rand(2700, 24)
    y = np.random.randint(0, 2, 2700)
    
    t0 = time.time()
    for i in range(100):
        model = XGBClassifier(n_estimators=100, max_depth=4, verbosity=0, n_jobs=1)
        model.fit(X, y)
    t1 = time.time()
    
    print(f"Time for 100 models: {t1 - t0:.2f} seconds")
    print(f"Estimated time for 1365 models (15 choose 4): {(t1 - t0) * 13.65 / 60:.2f} minutes")

if __name__ == "__main__":
    test_speed()
