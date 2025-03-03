# from configs.utils import get_config

# a = get_config(r"configs\UIT_VFSC\sentiment\BiGRU\s1\config_vipherv2_BiGRU_UIT_VFSC_sentiment.yaml")
# print(a.embedder)

import torch
print(torch.cuda.is_available())  # Kiểm tra xem có GPU khả dụng không
print(torch.version.cuda)         # Kiểm tra phiên bản CUDA mà PyTorch hỗ trợ
