import torch
import numpy as np
from typing import List
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# VAE 缩放因子（SD v1.4 标准值）
LATENT_SCALING = 0.18215


class DiffusionModel:
    """
    基于 diffusers 的扩散模型封装，兼容原有接口
    """

    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
    ):
        """
        Parameters
        ----------
        model_dir : Path
            模型根目录，包含 unet, vae, text_encoder, tokenizer 子目录
        device : torch.device
            计算设备
        """
        self.device = device
        
        print(f"Loading diffusion models from {model_dir}...")
        
        # 加载模型组件（使用 safetensors 和 float32）
        self.unet = UNet2DConditionModel.from_pretrained(
            model_dir / "unet",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32
        ).to(device).eval()
        
        self.vae = AutoencoderKL.from_pretrained(
            model_dir / "vae",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32
        ).to(device).eval()
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_dir / "text_encoder",
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32
        ).to(device).eval()
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_dir / "tokenizer",
            local_files_only=True
        )
        
        # 配置 DDIM 调度器
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
        # 冻结所有参数
        for component in [self.unet, self.vae, self.text_encoder]:
            for p in component.parameters():
                p.requires_grad = False
        
        # 预计算无条件 embeddings
        with torch.no_grad():
            uncond_input = self.tokenizer(
                "", padding="max_length", max_length=77, 
                truncation=True, return_tensors="pt"
            ).to(device)
            self.null_cond = self.text_encoder(uncond_input.input_ids)[0]
        
        print("✓ All models loaded successfully!")

    # ------------------------------------------------------------------
    # E(x): Image -> Latent
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码图像到潜在空间
        
        Parameters
        ----------
        image : torch.Tensor
            Shape (B, 3, H, W), range [-1, 1]
            
        Returns
        -------
        latent : torch.Tensor
            Shape (B, 4, h, w)
        """
        image = image.to(self.device)
        # 修复：diffusers 的 encode 返回 AutoencoderKLOutput 对象
        # 需要通过 .latent_dist 获取分布
        latent_dist = self.vae.encode(image).latent_dist
        latent = latent_dist.sample() * LATENT_SCALING
        return latent

    # ------------------------------------------------------------------
    # R_theta(z; xi): 批量随机重建
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct_latent_batch(
        self,
        latents: torch.Tensor,
        seeds: List[int],
        num_steps: int,
        strength: float = 1.0,  # ✅ 新增参数
    ) -> torch.Tensor:
        """
        批量 DDIM 重建
        
        Parameters
        ----------
        latents : torch.Tensor
            原始潜在向量，Shape (B, 4, h, w)
        seeds : List[int]
            随机种子列表
        num_steps : int
            DDIM 总步数
        strength : float
            噪声强度 (0.0-1.0)。1.0=完全重建，0.0=无噪声
        """
        B = latents.shape[0]
        latents = latents.to(self.device)
        
        # ✅ 设置调度器时间步
        self.scheduler.set_timesteps(num_steps, device=self.device)
        
        # ✅ 根据 strength 计算起始步数
        # strength=1.0 → start_step = num_steps（从最后开始）
        # strength=0.5 → start_step = num_steps * 0.5（从中间开始）
        start_step = int(num_steps * strength)
        
        # 截取时间步：只使用后面的部分
        # timesteps 是从大到小排列的 [999, 950, 900, ...]
        timesteps_to_use = self.scheduler.timesteps[-start_step:]
        
        outputs = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # ✅ 噪声添加：只在需要的时间步添加噪声
            if strength < 1.0:
                # 如果 strength < 1.0，从指定时间步开始添加噪声
                start_timestep = timesteps_to_use[0]
            else:
                # strength = 1.0，从最后一个时间步开始
                start_timestep = self.scheduler.timesteps[0]
            
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, start_timestep)
            
            # ✅ 反向去噪：只遍历选中的时间步
            current_latents = latents_noisy
            
            for t in timesteps_to_use:
                # 准备模型输入（无条件）
                latent_model_input = torch.cat([current_latents] * 2)
                t_batch = torch.tensor([t] * B * 2, device=self.device, dtype=torch.long)
                
                # 无条件 embeddings
                uncond = self.null_cond.repeat(B, 1, 1)
                
                # 预测噪声
                noise_pred = self.unet(
                    latent_model_input, t_batch, 
                    torch.cat([uncond, uncond], dim=0)  # 两个都是无条件
                ).sample
                
                # 不使用分类器引导
                noise_pred_uncond, _ = noise_pred.chunk(2)
                
                # 调度器步骤
                current_latents = self.scheduler.step(
                    noise_pred_uncond, t, current_latents
                ).prev_sample
            
            outputs.append(current_latents)
        
        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # (可选) Latent -> Image
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        解码潜在向量到图像空间
        
        Parameters
        ----------
        latent : torch.Tensor
            Shape (B, C, h, w)
            
        Returns
        -------
        image : torch.Tensor
            Shape (B, 3, H, W), range [-1, 1]
        """
        latent = latent / LATENT_SCALING
        # diffusers 的 decode 返回 DecoderOutput 对象
        image = self.vae.decode(latent).sample
        return image
