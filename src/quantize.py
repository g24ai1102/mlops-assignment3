import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

# Load trained sklearn model
sk_model = joblib.load("models/model.joblib")

# Extract weights
coef = sk_model.coef_
intercept = sk_model.intercept_

# Save unquantized params
unquant_params = {"coef": coef, "intercept": intercept}
joblib.dump(unquant_params, "models/unquant_params.joblib")

# Manual quantization
# def quantize(arr):
#     scale = 255.0 / (arr.max() - arr.min())
#     zp = -arr.min() * scale
#     q_arr = np.round(arr * scale + zp).astype(np.uint8)
#     return q_arr, scale, zp

# def quantize(arr):
#     arr_min = arr.min()
#     arr_max = arr.max()
#     if arr_max == arr_min:
#         # All values are the same (e.g., intercept), use default scale
#         scale = 1.0
#         zp = 0.0
#         q_arr = np.round(arr).astype(np.uint8)
#     else:
#         scale = 255.0 / (arr_max - arr_min)
#         zp = -arr_min * scale
#         q_arr = np.clip(np.round(arr * scale + zp), 0, 255).astype(np.uint8)
#         # q_arr = np.round(arr * scale + zp).astype(np.uint8)
#     return q_arr, scale, zp

def quantize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max == arr_min:
        q_arr = np.zeros_like(arr, dtype=np.uint8)
        return q_arr, arr_min, arr_max
    # Min-max scaling
    q_arr = np.round((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    return q_arr, arr_min, arr_max


coef_q, coef_scale, coef_zp = quantize(coef)
intercept_q, intercept_scale, intercept_zp = quantize(np.array([intercept]))

quant_params = {
    "coef_q": coef_q,
    "intercept_q": intercept_q,
    "coef_scale": coef_scale,
    "intercept_scale": intercept_scale,
    "coef_zp": coef_zp,
    "intercept_zp": intercept_zp,
}
joblib.dump(quant_params, "models/quant_params.joblib")

# Dequantize
# def dequantize(q_arr, scale, zp):
#     return (q_arr.astype(np.float32) - zp) / scale
def dequantize(q_arr, arr_min, arr_max):
    return q_arr.astype(np.float32) / 255 * (arr_max - arr_min) + arr_min


coef_dq = dequantize(coef_q, coef_scale, coef_zp)
intercept_dq = dequantize(intercept_q, intercept_scale, intercept_zp)[0]

# Define PyTorch model
class QuantizedModel(nn.Module):
    def __init__(self, weights, bias):
        super().__init__()
        self.linear = nn.Linear(len(weights), 1)
        # self.linear.weight.data = torch.tensor([weights], dtype=torch.float32)
        self.linear.weight.data = torch.tensor(np.array([weights]), dtype=torch.float32)
        self.linear.bias.data = torch.tensor([bias], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)

# Load data and run inference
X, y = fetch_california_housing(return_X_y=True)
X_tensor = torch.tensor(X, dtype=torch.float32)

model = QuantizedModel(coef_dq, intercept_dq)
y_pred = model(X_tensor).detach().squeeze().numpy()

# Evaluate
r2 = r2_score(y, y_pred)
print(f"Quantized Model R² Score: {r2:.4f}")
