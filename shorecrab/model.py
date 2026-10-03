"""ShoreCrab candidate-scoring inference graph."""
import math
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import View, feature_geometry, geometry_embedding

class CrossAttention(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must divide heads')
        self.heads = heads
        self.dim = dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.dim // self.heads).transpose(1, 2)

    def forward(self, query, memory, weights=None):
        q, k, v = (self._heads(self.q(query)), self._heads(self.k(memory)), self._heads(self.v(memory)))
        bias = None
        if weights is not None:
            if weights.ndim != 1 or len(weights) != memory.shape[1] or (not bool((weights > 0).any())):
                raise ValueError('attention requires at least one valid owned key')
            bias = torch.where(weights > 0, weights.float().clamp_min(1e-12).log(), float('-inf')).to(q.dtype)[None, None, None, :]
        if k.shape[0] != q.shape[0]:
            k, v = (k.expand(q.shape[0], -1, -1, -1), v.expand(q.shape[0], -1, -1, -1))
        z = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=bias, dropout_p=0)
        z = z.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.dim)
        return self.out(z)

class FFN(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.net(x)

class Resampler(nn.Module):

    def __init__(self, d, heads, max_queries, residual_mode='conditioned_query', bank_std=0.02, qcond_init='default'):
        super().__init__()
        if residual_mode not in {'conditioned_query', 'learned_queries'}:
            raise ValueError('unknown resampler residual mode')
        if qcond_init not in {'default', 'zero'}:
            raise ValueError('unknown qcond init')
        self.residual_mode = residual_mode
        self.bank = nn.Parameter(torch.randn(1, max_queries, d) * bank_std)
        self.qcond = nn.Linear(d, d)
        if qcond_init == 'zero':
            nn.init.zeros_(self.qcond.weight)
            nn.init.zeros_(self.qcond.bias)
        self.qnorm = nn.LayerNorm(d)
        self.mnorm = nn.LayerNorm(d)
        self.attention = CrossAttention(d, heads)
        self.ffn = FFN(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, memory, weights, qcls, n_queries, query_offset=0):
        if query_offset < 0 or n_queries < 1 or query_offset + n_queries > self.bank.shape[1]:
            raise ValueError('requested queries exceed resampler bank')
        base = self.bank[:, query_offset:query_offset + n_queries]
        q = base + self.qcond(qcls)[:, None, :]
        residual = q if self.residual_mode == 'conditioned_query' else base
        q = residual + self.attention(self.qnorm(q), self.mnorm(memory), weights)
        return self.norm(self.ffn(q))

class ChoiceBlock(nn.Module):

    def __init__(self, d, heads):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=0, batch_first=True)
        self.qnorm = nn.LayerNorm(d)
        self.mnorm = nn.LayerNorm(d)
        self.cross = CrossAttention(d, heads)
        self.ffn = FFN(d)

    def forward(self, x, valid, memory, memory_valid=None):
        y = self.norm(x)
        x = x + self.self_attn(y, y, y, key_padding_mask=~valid, need_weights=False)[0]
        x = x + self.cross(self.qnorm(x), self.mnorm(memory), memory_valid)
        return self.ffn(x) * valid[:, :, None]

class MCJev(nn.Module):

    def __init__(self, vision, text, dim=384, heads=6, global_queries=64, local_queries=32, fusion_layers=2, resampler_residual='conditioned_query', local_resampler='free', resampler_bank_std=0.02, resampler_qcond_init='default', summary_source='locals', overview_skip=False):
        super().__init__()
        if dim % 8 or dim % heads:
            raise ValueError('dim must be divisible by 8 and heads')
        if min(global_queries, local_queries) < 1:
            raise ValueError('queries must be positive')
        if local_resampler not in {'free', 'spatial_2x2'}:
            raise ValueError('unknown local resampler mode')
        if local_resampler == 'spatial_2x2' and local_queries != 32:
            raise ValueError('spatial_2x2 requires 4 regions x 8 queries')
        self.vision = vision
        self.text = text
        self.dim = dim
        self.global_queries = global_queries
        self.local_queries = local_queries
        self.local_resampler = local_resampler
        c16, c32 = vision.channels
        self.project16 = nn.Conv2d(c16, dim, 1)
        self.project32 = nn.Conv2d(c32, dim, 1)
        self.scale32 = nn.Parameter(torch.tensor(0.1))
        self.visual_norm = nn.LayerNorm(dim)
        if summary_source not in {'locals', 'overview'}:
            raise ValueError('unknown summary_source')
        self.overview_skip = nn.Linear(dim, dim) if overview_skip else None
        if overview_skip:
            nn.init.zeros_(self.overview_skip.weight)
            nn.init.zeros_(self.overview_skip.bias)
        self.summary_source = summary_source
        self.resampler = Resampler(dim, heads, max(global_queries, local_queries), residual_mode=resampler_residual, bank_std=resampler_bank_std, qcond_init=resampler_qcond_init)
        self.memory_type = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.fusion = nn.ModuleList([ChoiceBlock(dim, heads) for _ in range(fusion_layers)])
        self.scorer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.answerability = nn.Sequential(nn.Linear(4 * dim + 1, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def encode_view(self, pixels):
        f16, f32 = self.vision(pixels)
        f = self.project16(f16) + self.scale32 * F.interpolate(self.project32(f32), size=f16.shape[-2:], mode='bilinear', align_corners=False)
        return (self.visual_norm(f.flatten(2).transpose(1, 2)), f.shape[-2:])

    def _spatial_read(self, memory, weights, qcls, fh, fw):
        rows = torch.arange(fh, device=memory.device) * 2 // fh
        cols = torch.arange(fw, device=memory.device) * 2 // fw
        regions = (rows[:, None] * 2 + cols[None, :]).reshape(-1)
        outputs = []
        valid = []
        for region in range(4):
            region_keys = (regions == region) & (weights > 0)
            populated = bool(region_keys.any())
            if populated:
                z = self.resampler(memory[:, region_keys], weights[region_keys], qcls, 8, query_offset=region * 8)
            else:
                z = memory.new_zeros(memory.shape[0], 8, self.dim)
            outputs.append(z)
            valid.append(torch.full((8,), populated, dtype=torch.bool, device=memory.device))
        return (torch.cat(outputs, 1), torch.cat(valid))
    VIEW_CHUNK = 16

    def encode_views(self, views, fixed_chunk=True):
        """Encode same-size views (possibly from several scenes); returns per-view (tokens,(fh,fw)).
        Views go through the encoder in fixed chunks of VIEW_CHUNK, zero-padded: every distinct batch shape makes MPS
        keep another graph/heap (measured +6.9 GiB for 24 shapes), which exhausted memory mid-run. Pretrained BN
        statistics are frozen, so padding never affects real views."""
        device = next(self.parameters()).device
        x = torch.stack([v.pixels for v in views])
        out = []
        if not fixed_chunk:
            tokens, grid = self.encode_view(x.to(device))
            return [(tokens[j:j + 1], grid) for j in range(len(views))]
        for i in range(0, len(x), self.VIEW_CHUNK):
            chunk = x[i:i + self.VIEW_CHUNK]
            n = len(chunk)
            if n < self.VIEW_CHUNK:
                chunk = torch.cat((chunk, chunk.new_zeros(self.VIEW_CHUNK - n, *chunk.shape[1:])))
            tokens, grid = self.encode_view(chunk.to(device))
            out.extend(((tokens[j:j + 1], grid) for j in range(n)))
        return out

    def _read_view(self, pixels, owned, geometry, qcls, n_queries, spatial=False, encoded=None):
        tokens, (fh, fw) = self.encode_view(pixels) if encoded is None else encoded
        weights = F.adaptive_avg_pool2d(owned.float(), (fh, fw)).reshape(-1)
        if geometry.shape[0] != tokens.shape[1]:
            raise RuntimeError('encoder stride contract changed')
        emb = geometry_embedding(geometry, self.dim).to(tokens.dtype)
        memory = tokens + emb[None]
        if spatial:
            z, valid = self._spatial_read(memory, weights, qcls, fh, fw)
        else:
            z = self.resampler(memory, weights, qcls, n_queries)
            valid = torch.ones(n_queries, dtype=torch.bool, device=memory.device)
        weight_sum = weights.sum()
        summary = (tokens.float() * weights[None, :, None]).sum(1)
        return (z, summary, weight_sum, valid)

    def forward(self, views: Iterable[View], question: str, choices: list[str], encoded=None):
        if not 2 <= len(choices) <= 16:
            raise ValueError('2..16 choices required')
        device = next(self.parameters()).device
        q, a, amask = self.text.encode(question, choices, device)
        qcls = q.mean(dim=1)
        zs = []
        zvalid = []
        numer = None
        denom = None
        local_count = 0
        overview_count = 0
        overview = None
        for vi, view in enumerate(views):
            pixels = view.pixels[None].to(device)
            owned = view.owned[None].to(device)
            fh, fw = (pixels.shape[-2] // 16, pixels.shape[-1] // 16)
            geometry = feature_geometry(view, fh, fw).to(device)
            n = self.global_queries if view.kind == 'overview' else self.local_queries
            spatial = view.kind == 'local' and self.local_resampler == 'spatial_2x2'

            def read(p, o, g, c, nq=n, region_mode=spatial):
                return self._read_view(p, o, g, c, nq, region_mode)
            if encoded is not None:
                z, s, count, valid = self._read_view(None, owned, geometry, qcls, n, spatial, encoded[vi])
            else:
                z, s, count, valid = read(pixels, owned, geometry, qcls)
            zs.append(z)
            zvalid.append(valid)
            if view.kind != 'local' and self.overview_skip is not None:
                zs.append(self.overview_skip((s / count.clamp_min(1e-12)).to(z.dtype))[:, None, :])
                zvalid.append(torch.ones(1, dtype=torch.bool, device=z.device))
            if view.kind == 'local':
                local_count += 1
            else:
                overview_count += 1
            if view.kind == ('overview' if self.summary_source == 'overview' else 'local'):
                numer = s if numer is None else numer + s
                denom = count if denom is None else denom + count
            if view.kind != 'local':
                if encoded is not None:
                    overview = (encoded[vi][0], encoded[vi][1], F.adaptive_avg_pool2d(owned.float(), encoded[vi][1]).reshape(-1), view, qcls, geometry)
        if overview_count != 1 or (local_count < 1 and self.summary_source == 'locals'):
            raise ValueError('one overview and all planned native tiles are required')
        visual_valid = torch.cat(zvalid)
        visual = (torch.cat(zs, 1) + self.memory_type[0]) * visual_valid[None, :, None]
        memory = torch.cat((visual, q + self.memory_type[1]), 1)
        memory_valid = None
        if self.local_resampler == 'spatial_2x2':
            memory_valid = torch.cat((visual_valid, torch.ones(q.shape[1], dtype=torch.bool, device=device)))
        for block in self.fusion:
            a = block(a, amask, memory, memory_valid)
        h = a[:, 0]
        logits = self.scorer(h).squeeze(-1).float()
        summary = (numer / denom.clamp_min(1e-12)).to(q.dtype)
        joined = torch.cat((qcls, summary, h.mean(0, keepdim=True), h.max(0, keepdim=True).values, q.new_tensor([[math.log(len(choices))]])), 1)
        u = self.answerability(joined).reshape(()).float()
        return {'option_logits': logits, 'answerability_logit': u, 'processed_tiles': local_count, 'visual_tokens': visual.shape[1], 'valid_visual_tokens': visual_valid.sum(), 'overview': overview}
