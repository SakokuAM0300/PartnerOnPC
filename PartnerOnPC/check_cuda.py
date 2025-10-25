import torch

# CUDAが利用可能かチェックし、結果を表示
print(f"CUDA Available: {torch.cuda.is_available()}")

# どのデバイスが利用可能か知りたい場合は、以下も追加
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device Name: {torch.cuda.get_device_name(0)}")