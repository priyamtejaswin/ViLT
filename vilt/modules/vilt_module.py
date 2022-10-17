import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer as vit

from vilt.modules import heads, objectives, vilt_utils


class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config, text_embeddings, bert_config):
        super().__init__()
        self.save_hyperparameters()
        
        self.text_embeddings = text_embeddings
        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        # if config["loss_names"]["mlm"] > 0:
        #     self.mlm_score = heads.MLMHead(bert_config)
        #     self.mlm_score.apply(objectives.init_weights)

        # if config["loss_names"]["itm"] > 0:
        #     self.itm_score = heads.ITMHead(config["hidden_size"])
        #     self.itm_score.apply(objectives.init_weights)

        # if config["loss_names"]["mpp"] > 0:
        #     self.mpp_score = heads.MPPHead(bert_config)
        #     self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        # if self.hparams.config["loss_names"]["vqa"] > 0:
        vs = self.hparams.config["vqav2_label_size"]
        self.vqa_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
        )
        self.vqa_classifier.apply(objectives.init_weights)

        vilt_utils.set_metrics(self)
        self.current_tasks = []

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        imgkey = "image"

        text_ids = batch[f"text_ids"]
        text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        img = batch[imgkey]
        (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
        ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams.config["max_image_len"],
                mask_it=mask_image,
        )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)
        return cls_feats

        # ret = {
        #     # "text_feats": text_feats,
        #     # "image_feats": image_feats,
        #     "cls_feats": cls_feats,
        #     # "raw_cls_feats": x[:, 0],
        #     # "image_labels": image_labels,
        #     # "image_masks": image_masks,
        #     # "text_labels": text_labels,
        #     # "text_ids": text_ids,
        #     # "text_masks": text_masks,
        #     # "patch_index": patch_index,
        # }

        # return ret

    def forward(self, batch):

        logits = self.infer(batch)
        return self.vqa_classifier(logits)

        #ret = dict()
        #if len(self.current_tasks) == 0:
        #    res = self.infer(batch)
        #     print("about to return")
        #     ret.update(res)
        #     return ret

        # # Visual Question Answering
        # if "vqa" in self.current_tasks:
        #     ret.update(objectives.compute_vqa(self, batch))

        # return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
