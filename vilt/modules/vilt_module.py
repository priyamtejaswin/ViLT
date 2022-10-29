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
        cls_feats = self.pooler(x)
        return cls_feats

    def forward(self, batch):

        logits = self.infer(batch)
        return self.vqa_classifier(logits)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
