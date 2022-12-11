import numpy as np
import torch
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

config = BertConfig(torchscript=True)
embed = BertEmbeddings(config)
embed.eval()

x = torch.tensor(np.array([[101, 2003, 2023, 10733, 23566, 1029, 102]]))

scripted = torch.jit.trace(embed, x)
print(scripted)

