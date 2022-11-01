import torch
import copy
import os

# from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import objectives
from torchvision import transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer

from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

ASSETS_PATH = "HelloWorld/app/src/main/assets"
assert os.path.isdir(ASSETS_PATH)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    _config["test_only"] = True
    _config["load_path"] = "./weights/vilt_vqa.ckpt"
    assert os.path.exists(_config["load_path"]) and os.path.isfile(_config["load_path"])

    print(_config)
    print("Preparing batch for JIT tracing.")

    # url = "https://media.istockphoto.com/photos/joyful-dog-playing-with-whip-while-walking-on-green-field-picture-id1187003477?k=20&m=1187003477&s=612x612&w=0&h=fvUFuwvTZWEJjk8HUU80-zvaI4gg9szPGJ2RdASH72s="
    # text = "What is the dog doing in this picture?"
    # res = requests.get(url)
    image = Image.open("HelloWorld/app/src/main/assets/helmet.jpg").convert("RGB")
    image = transforms.ToTensor()(image).unsqueeze_(0)
    with open("HelloWorld/app/src/main/assets/helmet.txt") as fp:
        text = fp.read().strip()

    img = pixelbert_transform(size=384)(image)
    # img = img.unsqueeze(0)
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    batch = {"text": [text], "image": img}
    encoded = tokenizer(batch["text"])
    print(encoded)
    batch["text"] = torch.tensor(encoded["input_ids"])
    batch["text_ids"] = torch.tensor(encoded["input_ids"])
    batch["text_labels"] = torch.tensor(encoded["input_ids"])
    batch["text_masks"] = torch.tensor(encoded["attention_mask"])

    bert_config = BertConfig(
            vocab_size=_config["vocab_size"],
            hidden_size=_config["hidden_size"],
            num_hidden_layers=_config["num_layers"],
            num_attention_heads=_config["num_heads"],
            intermediate_size=_config["hidden_size"] * _config["mlp_ratio"],
            max_position_embeddings=_config["max_text_len"],
            hidden_dropout_prob=_config["drop_rate"],
            attention_probs_dropout_prob=_config["drop_rate"],
    )

    text_embeddings = BertEmbeddings(bert_config)
    text_embeddings.apply(objectives.init_weights)

    print("Creating ViLT model.")
    model = ViLTransformerSS(_config, text_embeddings, bert_config)
    model.setup("test")
    model.eval()

    print("Scripting.")
    traced_model = torch.jit.trace(model, batch)
    print("Optimizing.")
    opt_traced_model = optimize_for_mobile(traced_model)
    opt_traced_model._save_for_lite_interpreter(
        os.path.join(ASSETS_PATH, "optTracedVilt.ptl")
    )
    print("Done.")

    print("Scripting pixelbert transform.")
    traced_pixelbert = torch.jit.script(pixelbert_transform(384))
    traced_pixelbert._save_for_lite_interpreter(
        os.path.join(ASSETS_PATH, "optPixelbertTransform.ptl")
    )
    print("Done.")

    print("Assets saved to", ASSETS_PATH)
