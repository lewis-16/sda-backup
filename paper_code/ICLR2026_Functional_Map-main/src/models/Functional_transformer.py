import torch
import torch.nn as nn
import math

class TimePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def get(self, L): return self.pe[:L, :]

class QueryReconTransformerPruned(nn.Module):
    def __init__(self, func_emb_dim, fs, d_model=128, nhead=4,
                 num_enc_layers=4, num_dec_layers=4, dim_ff=512, dropout=0.1,
                 patch_ms=20.0, stride_ms=None):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.fs = fs
        self.patch_len = max(1, int(round((patch_ms/1000.0)*fs)))
        self.stride    = self.patch_len if stride_ms is None else max(1, int(round((stride_ms/1000.0)*fs)))

        # Conv tokenizer for signals
        self.conv_tok  = nn.Conv1d(1, d_model, kernel_size=self.patch_len, stride=self.stride, bias=False)
        # Functional (one-hot or embedding) projection
        self.func_proj = nn.Linear(func_emb_dim, d_model)
        # Concat + fuse
        self.fuse      = nn.Linear(2*d_model, d_model)

        # Type embeddings
        self.is_source = nn.Parameter(torch.randn(1,1,d_model)*0.02)
        self.is_query  = nn.Parameter(torch.randn(1,1,d_model)*0.02)

        # Time-only positional encoding
        self.tpos = TimePositionalEncoding(d_model)
        self.pre_ln_src = nn.LayerNorm(d_model)
        self.pre_ln_qry = nn.LayerNorm(d_model)

        # Encoder/Decoder
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_enc_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_dec_layers)

        # Head to waveform (per token → patch)
        self.head = nn.Linear(d_model, self.patch_len)

        # Learned query base per patch index
        self.query_base_len = 1024
        self.query_base = nn.Parameter(torch.randn(self.query_base_len, d_model)*0.02)

        # Optional gates to balance scales
        self.g_conv = nn.Parameter(torch.tensor(1.0))
        self.g_func = nn.Parameter(torch.tensor(1.0))

    def _enc_tokens(self, waves_src, fembs_src, pad_mask):
        """
        waves_src: (B, Csrc_max, T)
        fembs_src: (B, Csrc_max, F)
        pad_mask:  (B, Csrc_max)  True=PAD
        -> src_tokens: (B, Csrc_max*P, d), keymask: (B, Csrc_max*P), P
        """
        B, Csrc_max, T = waves_src.shape
        x = waves_src.view(B*Csrc_max, 1, T)
        conv = self.conv_tok(x)                            # (B*Csrc, d, P)
        P = conv.shape[-1]
        conv = conv.permute(0,2,1).view(B, Csrc_max, P, self.d_model)  # (B,Csrc,P,d)

        fproj = self.func_proj(fembs_src)                 # (B,Csrc,d)
        fexp  = fproj[:, :, None, :].expand_as(conv)      # (B,Csrc,P,d)

        cat   = torch.cat([self.g_conv*conv, self.g_func*fexp], dim=-1)
        fused = self.fuse(cat)                            # (B,Csrc,P,d)

        # time-only PE (same across channels for same patch index)
        pe_t = self.tpos.get(P)[None, None, :, :]         # (1,1,P,d)
        fused = self.pre_ln_src(fused + self.is_source + pe_t)

        src_tokens = fused.reshape(B, Csrc_max*P, self.d_model)
        keymask    = pad_mask[:, :, None].expand(B, Csrc_max, P).reshape(B, Csrc_max*P)  # True=PAD
        return src_tokens, keymask, P

    def _qry_tokens(self, femb_tgt, P, tgt_pad_mask=None):
        """
        femb_tgt:
          - (B, F): single target -> (B, P, d)
          - (B, K, F): multi-target -> (B, K*P, d) and returns (q, K, tgt_keymask_flat)
        tgt_pad_mask (optional): (B, K) True=PAD for multi-target
        """
        if femb_tgt.dim() == 2:
            B = femb_tgt.size(0)
            f_tgt = self.func_proj(femb_tgt)                  # (B,d)
            f_tgt = f_tgt[:, None, :].expand(B, P, self.d_model)

            # query base length guard
            if P <= self.query_base_len:
                q_base = self.query_base[:P, :]
            else:
                extra = self.query_base[-1:, :].expand(P - self.query_base_len, -1)
                q_base = torch.cat([self.query_base, extra], dim=0)
            q_base = q_base[None, :, :].expand(B, P, self.d_model)

            pe_t   = self.tpos.get(P)[None, :, :].expand(B, P, self.d_model)
            q_cat  = torch.cat([self.g_conv*q_base, self.g_func*f_tgt], dim=-1)
            q      = self.fuse(q_cat)
            q      = self.pre_ln_qry(q + self.is_query + pe_t)
            return q

        # Multi-target path: femb_tgt is (B, K, F)
        B, K, _ = femb_tgt.shape
        f_tgt = self.func_proj(femb_tgt)                      # (B, K, d)
        f_tgt = f_tgt[:, :, None, :].expand(B, K, P, self.d_model)  # (B, K, P, d)

        # query base
        if P <= self.query_base_len:
            q_base = self.query_base[:P, :]
        else:
            extra = self.query_base[-1:, :].expand(P - self.query_base_len, -1)
            q_base = torch.cat([self.query_base, extra], dim=0)
        q_base = q_base[None, None, :, :].expand(B, K, P, self.d_model)  # (B,K,P,d)

        pe_t = self.tpos.get(P)[None, None, :, :].expand(B, K, P, self.d_model)
        q_cat = torch.cat([self.g_conv*q_base, self.g_func*f_tgt], dim=-1)     # (B,K,P,2d)
        q = self.fuse(q_cat)                                                   # (B,K,P,d)
        q = self.pre_ln_qry(q + self.is_query + pe_t)                          # (B,K,P,d)

        # Flatten K*P for the decoder
        q = q.reshape(B, K*P, self.d_model)                                    # (B,KP,d)

        # Build tgt_key_padding_mask for decoder if provided (repeat over P)
        tgt_keymask_flat = None
        if tgt_pad_mask is not None:
            # tgt_pad_mask: (B,K) True=PAD -> repeat across P
            tgt_keymask_flat = tgt_pad_mask[:, :, None].expand(B, K, P).reshape(B, K*P)  # (B,KP)
        return q, K, tgt_keymask_flat

    def forward(self, waves_src, fembs_src, pad_mask, femb_tgt, tgt_pad_mask=None):
        """
        waves_src: (B, Csrc_max, T)   sources only
        fembs_src:(B, Csrc_max, F)
        pad_mask: (B, Csrc_max)       True=PAD
        femb_tgt:
          - (B, F)         -> single target      → y_hat: (B, Tm)
          - (B, K, F)      -> multi-target (pad) → y_hat: (B, K, Tm), using tgt_pad_mask (B,K) True=PAD

        """
        src_tokens, keymask, P = self._enc_tokens(waves_src, fembs_src, pad_mask)
        memory = self.encoder(src_tokens, src_key_padding_mask=keymask)         # (B, S_src, d)

        if femb_tgt.dim() == 2:
            # single target
            qry = self._qry_tokens(femb_tgt, P)                                 # (B, P, d)
            dec = self.decoder(qry, memory, memory_key_padding_mask=keymask)    # (B, P, d)
            patches = self.head(dec)                                            # (B, P, patch_len)
            y_hat = patches.reshape(patches.size(0), -1)                        # (B, Tm)
            return y_hat
        else:
            # multi-target
            qry, K, tgt_keymask_flat = self._qry_tokens(femb_tgt, P, tgt_pad_mask)  # (B, KP, d)
            dec = self.decoder(
                qry, memory,
                memory_key_padding_mask=keymask,
                tgt_key_padding_mask=tgt_keymask_flat
            )                                                                    # (B, KP, d)
            patches = self.head(dec)                                             # (B, KP, patch_len)
            patches = patches.view(patches.size(0), K, P, self.patch_len)       # (B, K, P, L)
            y_hat = patches.reshape(patches.size(0), K, -1)                     # (B, K, Tm)
            return y_hat


