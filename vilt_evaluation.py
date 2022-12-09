import torch
import copy
import time
import requests
import io
import numpy as np
# from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils
import vilt.modules.vision_transformer as vit
# import timm.models.vision_transformer as vit

from torchvision import transforms
import json
import urllib

from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer

import requests
import torch
import os

from PIL import Image

from vilt.config import ex
from vilt.modules import ViLTransformerSS

from tqdm import tqdm

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    _config["test_only"] = True
    _config["load_path"] = "./weights/vilt_vqa.ckpt"
    assert os.path.exists(_config["load_path"]) and os.path.isfile(_config["load_path"])

    loss_names = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 1,
        "imgcls": 0,
        "nlvr2": 0,
        "irtr": 0,
        "arc": 0,
    }

    print(_config)

    with urllib.request.urlopen(
        "https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json"
    ) as url:
        id2ans = json.loads(url.read().decode())

    url = "https://media.istockphoto.com/photos/joyful-dog-playing-with-whip-while-walking-on-green-field-picture-id1187003477?k=20&m=1187003477&s=612x612&w=0&h=fvUFuwvTZWEJjk8HUU80-zvaI4gg9szPGJ2RdASH72s="
    text = "What is the dog doing in this picture?"
    res = requests.get(url)
    image = Image.open(io.BytesIO(res.content)).convert("RGB")
    image = transforms.ToTensor()(image).unsqueeze_(0)

    pbtr = pixelbert_transform(size=384)
    img = pbtr(image)
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
    text_embeddings.eval()

    scripted_te = torch.jit.trace(text_embeddings, torch.tensor(
        np.array([[101, 2003, 2023, 10733, 23566, 1029, 102]])
    ))

    visiontransformer = getattr(vit, _config["vit"])(
        pretrained=False, config=_config
    )

    # scripted_vt = torch.jit.script(visiontransformer)
    
    # model = ViLTransformerSS(_config, text_embeddings, visiontransformer)
    model = ViLTransformerSS(_config, scripted_te, visiontransformer)
    print("Created model!")
    model.setup("test")
    model.eval()

    logits = model(batch)
    print(logits)
    answer = id2ans[str(logits.argmax().item())]
    print(answer)
    
    # trace_model = torch.jit.trace(model, batch)
    trace_model = torch.jit.script(model)
    logits = trace_model(batch)
    print(logits)
    answer = id2ans[str(logits.argmax().item())]
    print(answer)
    print()

    # TESTS
    for url, question in [
            (
                "https://vault.si.com/.image/ar_1:1%2Cc_fill%2Ccs_srgb%2Cfl_progressive%2Cq_auto:good%2Cw_1200/MTY5MDk4NzMyODU4NzEzNTYz/si-vault_michael-jordan4jpg.jpg",
                "What game is being played?"
            ),
            (
                "https://media.istockphoto.com/id/1193587060/photo/modern-home-interior-in-trendy-colors-of-the-year-2020.jpg?s=612x612&w=0&k=20&c=iANTpyGwY5VqKbzm7ZGTERSkokSkI8xMb_jo4uCK3oQ=",
                "What is the color of the couch?"
            ),
            (
                "https://cdn.appuals.com/wp-content/uploads/2022/04/monitor-turned-off-automatically.jpg",
                "Is there anything on the screen?"
            )
        ]:
            res = requests.get(url)
            image = Image.open(io.BytesIO(res.content)).convert("RGB")
            image = transforms.ToTensor()(image).unsqueeze_(0)

            test_img = pbtr(image)
            batch = {"text": [question], "image": test_img}
            encoded = tokenizer(batch["text"])

            batch["text"] = torch.tensor(encoded["input_ids"])
            batch["text_ids"] = torch.tensor(encoded["input_ids"])
            batch["text_labels"] = torch.tensor(encoded["input_ids"])
            batch["text_masks"] = torch.tensor(encoded["attention_mask"])

            logits = model(batch)
            print(id2ans[str(logits.argmax().item())])
            logits = trace_model(batch)
            print(id2ans[str(logits.argmax().item())])
            print()

    answers_base = []
    answers_traced = []

    with open('../vqa2eval/v2_OpenEnded_mscoco_val2014_questions.json') as f:
        data = json.load(f)
        questions = data['questions']

        for question in tqdm(questions):
            # try:
            image_id = question['image_id']
            text = question['question']
            question_id = question['question_id']
            image_path = '../vqa2eval/val2014/COCO_val2014_000000' + str(image_id).rjust(6, '0') + '.jpg'
            image = Image.open(image_path).convert("RGB")
            image = transforms.ToTensor()(image).unsqueeze_(0)
            img = pbtr(image)
            batch = {"text": [text], "image": img}
            encoded = tokenizer(batch["text"])
            #print(encoded)
            batch["text"] = torch.tensor(encoded["input_ids"])
            batch["text_ids"] = torch.tensor(encoded["input_ids"])
            batch["text_labels"] = torch.tensor(encoded["input_ids"])
            batch["text_masks"] = torch.tensor(encoded["attention_mask"])

            # logits = model(batch)
            #print(logits)
            # answer = id2ans[str(logits.argmax().item())]
            #print(answer)
            # answers_base.append(
                # {"answer": answer, "question_id": question_id}
            # )
            # except:
            #     answers_base.append(
            #         {"answer": '0', "question_id": question_id}
            #     )

            logits = trace_model(batch)
            # print(logits)
            answer = id2ans[str(logits.argmax().item())]
            # print(answer)
            answers_traced.append(
                {"answer": answer, "question_id": question_id}
            )

        with open('result-test-dev.json', 'w') as result_file:
            json.dump(answers_base, result_file)

        with open('result-traced-test-dev.json', 'w') as result_file_traced:
            json.dump(answers_traced, result_file_traced)

