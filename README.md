# Mobile ViLT

*This repo is a work-in-progress!*

# Introduction

This repo attempts to port the **ViLT** model -- described in the ICML 2021 (long talk) paper: "[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)" -- for mobile devices (modern Android systems). It forks and builds over the original repository, found [here](https://github.com/dandelin/ViLT). There is a [Huggingface module for ViLT (with a QA head)](https://huggingface.co/docs/transformers/model_doc/vilt#transformers.ViltForQuestionAnswering), but the interface and APIs are bloated, due to cross-compatibility reasons.

This effort was inspired by the [2022 MobiVQA paper](https://awk.ai/assets/mobivqa.pdf) which reportedly deploys a modified LXMERT model for VQA on mobile hardware (Nvidia Jetson and Google Pixel). Unfortunately, the [MobiVQA repository](https://github.com/SBUNetSys/MobiVQA/issues/1) contains no instructions on running (or replicating) the results. Important parts of the code (like configs, data setup scripts, etc) are missing, including perhaps the most crucial part of modifying the model and pre-processors for [Torchscript](https://pytorch.org/tutorials/recipes/torchscript_inference.html) compatibility. 

Our goal is to implement the optimizations discussed in MobiVQA on top of the ViLT architecture, while also ensuring that the model and pre-processors are Torchscript compatible.

# Resources on Torch JIT

* Working example of scripting a Huggingface model *that is designed to be scriptable* -- <https://huggingface.co/docs/transformers/main/en/torchscript>
* A short introduction to `script` vs `trace` functionality, and handling some edge cases -- <https://ppwwyyxx.com/blog/2022/TorchScript-Tracing-vs-Scripting/>
* Scripting a "complex" PyTorch chatbot model -- <https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html>
* Some utilities in `detectron2` which can help in scripting -- <https://github.com/facebookresearch/detectron2/blob/e091a07ef573915056f8c2191b774aad0e38d09c/detectron2/export/flatten.py#L186-L208>
* Another example of scripting a "complex" PyTorch translation model -- <https://colab.research.google.com/drive/1HiICg6jRkBnr5hvK2-VnMi88Vi9pUzEJ>
* (and finally) PyTorch documentation on torchscript -- <https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html#introduction-to-torchscript>

# Android app

The `HelloWorld` directory links to android app submodule -- <https://github.com/priyamtejaswin/HelloWorld>. Refer to **Setup, Eval** below before building the app. The app uses a Java implementation of the BERT pre-processor for the question (adapted from [this repo](https://github.com/huggingface/tflite-android-transformers/blob/master/bert/src/main/java/co/huggingface/android_transformers/bertqa/tokenization/FullTokenizer.java)). The image is pre-processed using PyTorch code, exported via Torchscipt. These are passed to the ViLT model (also exported using Torchscipt).

# Setup

Ensure you have the [ViLT VQA2 checkpoint](https://github.com/dandelin/ViLT/releases/download/200k/vilt_vqa.ckpt) downloaded to the `./weights` directory.

Also ensure that the model is in `test` mode.

Check that the Android App source (`HelloWorld`) is at the latest commit.

* `git submodule update --init`
* `git submodule update --remote --merge`

If you have the latest changes, the last command should not show any updates in `git status`.

To check functionality and correctness, run `python vilt_jit.py`

To prepare Torchscript files for the app, run `generate_assets.py`. This will save all Torchscript assets to `HelloWorld/app/src/main/assets/`.

ViLT is fine-tuned on VQA2 as a classification problem. Download the dict to assets -- <https://github.com/dandelin/ViLT/releases/download/200k/vqa_dict.json>

# Eval (VQA2)

To evaluate the model (base, scripted, or quantized) first setup the following in a directory OUTSIDE base `../vqa2eval/`

```bash
mkdir ../vqa2eval/
# Download validation data from the VQA website -- https://visualqa.org/download.html
cd ../vqa2eval/
wget --content-disposition https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
wget --content-disposition http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```

Now, run `python vilt_evaluation.py` and wait for 20 hrs.

# Additional tests
```python
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

        model(batch)
        trace_model(batch)
```

-- Priyam, Rishubh, Bi
