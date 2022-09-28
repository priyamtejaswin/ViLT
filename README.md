# Mobile ViLT

*This repo is a work-in-progress!*

---

This repo attempts to port the **ViLT** model -- described in the ICML 2021 (long talk) paper: "[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)" -- for mobile devices (modern Android systems). It forks and builds over the original repository, found [here](https://github.com/dandelin/ViLT).

This effort was inspired by the [2022 MobiVQA paper](https://awk.ai/assets/mobivqa.pdf) which reportedly deploys a modified LXMERT model for VQA on mobile hardware (Nvidia Jetson and Google Pixel). Unfortunately, the [MobiVQA repository](https://github.com/SBUNetSys/MobiVQA/issues/1) contains no instructions on running (or replicating) the results. Important parts of the code (like configs, data setup scripts, etc) are missing, including perhaps the most crucial part of modifying the model and pre-processors for [Torchscript](https://pytorch.org/tutorials/recipes/torchscript_inference.html) compatibility. 


Our goal is to implement the optimizations discussed in MobiVQA on top of the ViLT architecture, while also ensuring that the model and pre-processors are Torchscript compatible.

-- Priyam, Rishubh, Bi