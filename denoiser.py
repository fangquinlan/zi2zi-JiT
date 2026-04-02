import torch
import torch.nn as nn
import torch.nn.functional as F
from model_jit import JiT_models


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            num_fonts=args.num_fonts,
            num_chars=args.num_chars,
            use_char_embedding=getattr(args, "use_char_embedding", False),
            num_style_refs=getattr(args, "num_style_refs", 1),
            style_ref_mode=getattr(args, "style_ref_mode", "single"),
            use_ids_conditioning=getattr(args, "use_ids_conditioning", False),
            ids_vocab_size=getattr(args, "ids_vocab_size", 0),
            ids_max_len=getattr(args, "ids_max_len", 0),
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num
        self.num_fonts = args.num_fonts
        self.num_chars = args.num_chars

        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.binary_loss_weight = getattr(args, "binary_loss_weight", 0.0)
        self.edge_loss_weight = getattr(args, "edge_loss_weight", 0.0)
        self.projection_loss_weight = getattr(args, "projection_loss_weight", 0.0)
        self.char_loss_weight = getattr(args, "char_loss_weight", 0.0)
        self.ids_loss_weight = getattr(args, "ids_loss_weight", 0.0)
        self.char_consistency_head = nn.Linear(self.net.hidden_size, args.num_chars) if self.char_loss_weight > 0 else None
        ids_vocab_size = int(getattr(args, "ids_vocab_size", 0))
        self.ids_consistency_head = nn.Linear(self.net.hidden_size, ids_vocab_size + 1) if self.ids_loss_weight > 0 and ids_vocab_size > 0 else None
        self.last_loss_breakdown = {}

    def drop_labels(self, labels):
        font_labels, char_labels, style_images, content_images = labels
        batch_size = font_labels.shape[0]
        device = font_labels.device

        drop = torch.rand(batch_size, device=device) < self.label_drop_prob

        font_out = torch.where(drop, torch.full_like(font_labels, self.num_fonts), font_labels)
        char_out = torch.where(drop, torch.full_like(char_labels, self.num_chars), char_labels)

        drop_expanded_style = drop.view(batch_size, *([1] * (style_images.ndim - 1))).expand_as(style_images)
        drop_expanded_content = drop.view(batch_size, *([1] * (content_images.ndim - 1))).expand_as(content_images)

        style_out = torch.where(drop_expanded_style, torch.ones_like(style_images), style_images)
        content_out = torch.where(drop_expanded_content, torch.ones_like(content_images), content_images)

        return (font_out, char_out, style_out, content_out)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels):
        _, char_labels, _, _ = labels
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels_dropped)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        diffusion_loss = (v - v_pred) ** 2
        diffusion_loss = diffusion_loss.mean(dim=(1, 2, 3)).mean()

        aux_binary = x_pred.new_zeros(())
        aux_edge = x_pred.new_zeros(())
        aux_projection = x_pred.new_zeros(())
        aux_char = x_pred.new_zeros(())
        aux_ids = x_pred.new_zeros(())

        if self.binary_loss_weight > 0 or self.edge_loss_weight > 0 or self.projection_loss_weight > 0:
            pred_ink = self._to_ink_map(x_pred)
            target_ink = self._to_ink_map(x)
            if self.binary_loss_weight > 0:
                aux_binary = F.l1_loss(pred_ink, target_ink)
            if self.edge_loss_weight > 0:
                aux_edge = F.l1_loss(self._sobel_edges(pred_ink), self._sobel_edges(target_ink))
            if self.projection_loss_weight > 0:
                aux_projection = self._projection_consistency_loss(pred_ink, target_ink)

        if self.char_consistency_head is not None or self.ids_consistency_head is not None:
            semantic_feat = self.net.y_embedder.content_encoder(x_pred)
            if self.char_consistency_head is not None:
                char_logits = self.char_consistency_head(semantic_feat).float()
                aux_char = F.cross_entropy(char_logits, char_labels)
            if self.ids_consistency_head is not None:
                ids_targets = self.net.y_embedder.lookup_ids_bow(char_labels)
                ids_logits = self.ids_consistency_head(semantic_feat).float()
                aux_ids = F.binary_cross_entropy_with_logits(ids_logits[:, 1:], ids_targets[:, 1:])

        total_loss = (
            diffusion_loss
            + self.binary_loss_weight * aux_binary
            + self.edge_loss_weight * aux_edge
            + self.projection_loss_weight * aux_projection
            + self.char_loss_weight * aux_char
            + self.ids_loss_weight * aux_ids
        )
        self.last_loss_breakdown = {
            "diffusion_loss": float(diffusion_loss.detach().item()),
            "binary_loss": float(aux_binary.detach().item()),
            "edge_loss": float(aux_edge.detach().item()),
            "projection_loss": float(aux_projection.detach().item()),
            "char_loss": float(aux_char.detach().item()),
            "ids_loss": float(aux_ids.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }

        return total_loss

    def _to_ink_map(self, image):
        gray = (image + 1.0) * 0.5
        gray = gray.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        return 1.0 - gray

    def _sobel_edges(self, ink):
        kernel_x = ink.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        kernel_y = ink.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        grad_x = F.conv2d(ink, kernel_x, padding=1)
        grad_y = F.conv2d(ink, kernel_y, padding=1)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def _projection_consistency_loss(self, pred_ink, target_ink):
        pred_rows = pred_ink.mean(dim=3)
        pred_cols = pred_ink.mean(dim=2)
        target_rows = target_ink.mean(dim=3)
        target_cols = target_ink.mean(dim=2)
        return F.l1_loss(pred_rows, target_rows) + F.l1_loss(pred_cols, target_cols)

    def get_last_loss_breakdown(self):
        return dict(self.last_loss_breakdown)

    @torch.no_grad()
    def generate(self, labels):
        font_labels, char_labels, style_images, content_images = labels
        device = font_labels.device
        bsz = font_labels.size(0)
        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)
        sampling_context = self._prepare_sampling_context(labels)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        elif self.method == "ab2":
            return self._ab2_generate(z, timesteps, sampling_context)
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, sampling_context)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], sampling_context)
        return z

    @torch.no_grad()
    def _prepare_sampling_context(self, labels):
        font_labels, char_labels, _, _ = labels

        cond = self.net.y_embedder.encode(labels)
        null_labels = (
            torch.full_like(font_labels, self.num_fonts),
            torch.full_like(char_labels, self.num_chars),
            None,
            None,
        )
        uncond = self.net.y_embedder.encode(null_labels)

        return {
            "cond": cond,
            "uncond": uncond,
        }

    @torch.no_grad()
    def _forward_sample(self, z, t, sampling_context):
        cond = sampling_context["cond"]
        uncond = sampling_context["uncond"]

        # conditional
        x_cond = self.net.forward_with_conditioning(z, t.flatten(), cond)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net.forward_with_conditioning(z, t.flatten(), uncond)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, sampling_context):
        v_pred = self._forward_sample(z, t, sampling_context)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, sampling_context):
        v_pred_t = self._forward_sample(z, t, sampling_context)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, sampling_context)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _ab2_generate(self, z, timesteps, sampling_context):
        if self.steps <= 1:
            return self._euler_step(z, timesteps[0], timesteps[1], sampling_context)

        v_prev = self._forward_sample(z, timesteps[0], sampling_context)
        z = z + (timesteps[1] - timesteps[0]) * v_prev

        for i in range(1, self.steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            v_curr = self._forward_sample(z, t, sampling_context)
            z = z + (t_next - t) * (1.5 * v_curr - 0.5 * v_prev)
            v_prev = v_curr

        return z

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
