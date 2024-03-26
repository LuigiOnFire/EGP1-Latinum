"""
Based upon the nanoGPT and miniGPT implementatinos of Andrej Karpathy
"""
import math
from torch import nn

class MultiLevelPerceptron(nn.Module):
    def __init__(self, config):
        # first feedforward layer
        self.ff1 = nn.Linear(config.embedding_dim, 4*config.embedding_dim, bias=config.bias)
        self.gelu = nn.GELU()
        # second feedforward layer
        self.ff2 = nn.Linear(config.embedding_dim, 4*config.embedding_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        x = self.ff1(x)
        x = self.gelu(x)
        x = self.ff2(x)
        x = self.dropout(x)

        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.attn_head_count == 0, f"Embedding dimension ({config.embedding_dim}) must be divisible by number of attention heads ({config.attn_head_count})"
        self.attn_lin1 = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.attn_lin2 = nn.Lienar(config.embedding_dim, self.embedding_dim)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

        self.attn_head_num = config.attn_head_num
        self.embedding_dim = config.embedding_dim
        self.flash_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash_attn:
            print("Flash attention is not available. Check that Pytorch 2.0 or above is being used.")
            self.register_buffer("bias",
                torch.trill(
                   torch.ones(config.block_length, config.block_length)
                   .view(1, 1, config.block_length, config.block_length)
                )
            )

    def forward(self, x):
        # b = batch size
        # l = sequence length
        # d = embedding dimension
        b, l, d = x.size()

        qkv = self.attn_lin1(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)
        q = qview(b, l, self.attn_head_num, d // self.attn_head_num).transpose(1, 2)
        k = kview(b, l, self.attn_head_num, d // self.attn_head_num).transpose(1, 2)
        v = vview(b, l, self.attn_head_num, d // self.attn_head_num).transpose(1, 2)

        if self.flash_attn:
            out = torch.nn.functional.scaled_dot_product_attention(
                                                            q, k, v, \
                                                            atten_mask = None, \
                                                            dropout=self.attn_dropout if self.training else 0, \
                                                            is_causal=true \
                                                            )

        else:
            out = q @ k.transpose(-1, -2)
            out = out * (1.0 / math.sqrt(k.size(-1)))
            out = torch.masked_fill(mask(self.bias[:,:,:T,:T], float('-inf'))
            out = nn.functional.softmax(out, dim=-1)
            out = self.attn_dropout(out)
            out = out @ v

        out = out.transpose(1, 2).continguous().view(b, l, d)

        out = self.attn_lin2(output)
        out = self.resid_dropout(output)

        return output

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm1 = nn.Layernorm(config.embedding_dim,  config.bias)
        self.causal_self_attn = CausalSelfAttention(config)
        self.layernorm2 = nn.Layernorm(config.embedding_dim,  config.bias)
        self.mlp = MultiLevelPerceptron(config)

    def forward(self, x):
        x = self.layernorm1(x)
        x = self.causal_self_attn(x) + x
        x = self.layernorm(x)
        x = self.mlp(x) + x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # check here to make sure our relevant config stats are valid
        assert config.token_count is not None
        assert config.embedding_dim is not None
        assert config.block_length is not None
        assert config.embed_dropout is not None
        assert config.layer_count is not None
        
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                tkn_embd = nn.Embedding(config.token_count, config.embedding_dim),
                pos_embd = nn.Embedding(config.block_length, config.embedding_dim),
                dropout = nn.Dropout(config.embed_dropout),
                blocks = nn.ModuleList([Block(config) for _ in range(config.layer_count)),
                lyr_nrm = nn.LayerNorm(config.embedding_dim, bias=config.bias)
            )
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.token_count, bias=False)

        self.apply(self._init_weights)
        
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config.layer_count))

        param_count = sum(param.numel for param in self.transformer.parameters())
        print(f"This instantation of EGP has {param_count} parameters.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self, input_block, trgt_tkn):
        device = input_block.device()
        b, h = input_block.dims()
        assert h < self.config.block_length, f"Input token list too long: we received {h} and our maximum is {self.config.block_length."
        pos_seq = torch.arrange(0, h)

        # transformer forward here
        embds_t = self.transformer.tkn_embd(input_block)
        embds_p = self.transformer.pos_embd(pos_seq)
        embds_final = embds_t + embds_p
        out = dropout(embds_final) 
        for block in self.transformer.blocks:
            out = block(out)

        out = self.tranformer.lyr_nrm(out)

        logit = self.lm_head(out)

        return logit
        

    def loss(self, logit, trgt):
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), trgt, ignore_index=-1)
        return loss

    def 

    def configure_optimizers(self, optim_config):
        # This enables weight decay for certain parameters in our optimizer
        # See: https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab

        # Nab every param in our model, then filter out the ones that require grad
        param_dict = {name: param for name,param in self.named_parameters()}
        param_dict = {name: param for name,param in param_dict.items() if param.requires_grad}

        # Stuff all our remaining parameters into two categories based how many dimensions they have
        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        nodecay_params = [param, name, param, in param_dict.items() if param.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(param.numel() for param in decay_params)
        num_nodecay_params = sum(param.numel() for param in nodecay_params)
        print(f"We enable decay for {len(decay_params)} tensors.")
        print(f"These tensors with decay have {num_decay_params:.} parameters.")
        print(f"We disable decay for {len(nodecay_params)} tensors.")
        print(f"These tensors without decay have {num_nodecay_params:.} parameters.")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=Tru) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=config.learning_rate, 
                                      betas=config.betas, 
                                      **extra_args)
        print("Is our AdamW fused? {use_fused}")

        return optimizer

