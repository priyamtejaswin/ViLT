import torch
import copy
import os
from PIL import Image
import urllib
import json

# from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from torchvision import transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
import vilt.modules.vision_transformer as vit
from vilt.config import ex
from vilt.modules import ViLTransformerSS
from vilt.modules import objectives

ASSETS_PATH = "HelloWorld/app/src/main/assets"
assert os.path.isdir(ASSETS_PATH)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    _config["test_only"] = True
    _config["load_path"] = "./weights/vilt_vqa.ckpt"
    assert os.path.exists(_config["load_path"]) and os.path.isfile(_config["load_path"])

    with urllib.request.urlopen(
        "https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json"
    ) as url:
        id2ans = json.loads(url.read().decode())

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

    visiontransformer = getattr(vit, _config["vit"])(
        pretrained=False, config=_config
    )

    print("Creating ViLT model.")
    model = ViLTransformerSS(_config, text_embeddings, visiontransformer)
    model.setup("test")
    model.eval()

    print("Scripting.")
    traced_model = torch.jit.script(model)

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

    # Running tests ...
    print("Testing...")
    for f in os.listdir(ASSETS_PATH):
        if f.endswith(".jpg"):
            image = Image.open(os.path.join(ASSETS_PATH, f)).convert("RGB")
            image = transforms.ToTensor()(image).unsqueeze_(0)

            question = f.split('.')[0] + '.txt'
            with open(os.path.join(ASSETS_PATH, question)) as fp:
                text = fp.read().strip()

            img = traced_pixelbert(image)
            batch = {"text": [text], "image": img}

            tokenizer = get_pretrained_tokenizer(_config["tokenizer"])
            encoded = tokenizer(batch["text"])

            batch["text"] = torch.tensor(encoded["input_ids"])
            batch["text_ids"] = torch.tensor(encoded["input_ids"])
            batch["text_labels"] = torch.tensor(encoded["input_ids"])
            batch["text_masks"] = torch.tensor(encoded["attention_mask"])

            print("Test:", f)
            print("Question", text)
            logits = traced_model(batch)
            answer = id2ans[str(logits.argmax().item())]
            print("Answer:", answer)

    print("Assets saved to", ASSETS_PATH)
