import torch, torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model


class VLMDetector(nn.Module):
    """
    CLIP-ViT-B/32 + LoRA  (僅視覺編碼器最後兩層注入)
    輸出 512-D embedding，再接 1-unit sigmoid classifier。
    """
    def __init__(self, lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()

        # -------- 1. 下載 / 載入 CLIP --------
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # -------- 2. 注入 LoRA  ----------
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        self.clip.vision_model = get_peft_model(self.clip.vision_model, lora_cfg)

        # 先凍結全部，再放行 LoRA 權重
        for p in self.clip.parameters():
            p.requires_grad = False
        for n, p in self.clip.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        # -------- 3. 分類器 (512 → 1) --------
        self.embed_dim = self.clip.config.projection_dim          # 512
        self.classifier = nn.Linear(self.embed_dim, 1)

    # ------------------------------------------
    # 前向：PIL list → pixel_values → image embed → logit
    # ------------------------------------------
    def forward(self, pixel_values):
        # image_features: [B, 512]
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(image_features)                   # [B, 1]
        return logits.squeeze(-1)                                  # [B]

    # ------------------------------------------
    # 將 list[PIL.Image] → model 所需 tensor
    # ------------------------------------------
    def preprocess(self, pil_images, device):
        """
        Args
        ----
        pil_images : list of PIL.Image
        device     : torch.device
        Returns
        -------
        pixel_values : FloatTensor [B, 3, 224, 224]
        """
        inputs = self.processor(images=pil_images,
                                return_tensors="pt",
                                padding=True)
        return inputs["pixel_values"].to(device)
