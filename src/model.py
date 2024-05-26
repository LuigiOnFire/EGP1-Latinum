"""
Based upon the nanoGPT and miniGPT implementatinos of Andrej Karpathy
"""
import inspect
import math
import torch

class MultiLevelPerceptron(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # first feedforward layer
        self.ff1 = torch.nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"], bias=config["bias"])
        self.gelu = torch.nn.GELU()
        # second feedforward layer
        self.ff2 = torch.nn.Linear(4 * config["embedding_dim"], config["embedding_dim"], bias=config["bias"])
        self.dropout = torch.nn.Dropout(config["resid_dropout"])

    def forward(self, x):
        x = self.ff1(x)
        x = self.gelu(x)
        x = self.ff2(x)
        x = self.dropout(x)

        return x

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["embedding_dim"] % config["attn_head_count"] == 0, f"Embedding dimension ({config['embedding_dim']}) must be divisible by number of attention heads ({config['attn_head_count']})"
        self.attn_lin1 = torch.nn.Linear(config["embedding_dim"], 3 * config["embedding_dim"])
        self.attn_lin2 = torch.nn.Linear(config["embedding_dim"], config["embedding_dim"])

        self.attn_dropout = torch.nn.Dropout(config["attn_dropout"])
        self.resid_dropout = torch.nn.Dropout(config["resid_dropout"])
        self.attn_dropout_factor = config["attn_dropout"]

        self.attn_head_count = config["attn_head_count"]
        self.embedding_dim = config["embedding_dim"]

        # turns out flash attention not enabled on windows yet, unlucky        
        self.flash_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')        
        if not self.flash_attn:
            print("Flash attention is not available. Check that Pytorch 2.0 or above is being used.")
            self.register_buffer("bias",
                torch.tril(
                   torch.ones(config["block_length"], config["block_length"])
                   .view(1, 1, config["block_length"], config["block_length"])
                )
            )

    def forward(self, x):
        # b = batch size
        # l = sequence length
        # d = embedding dimension
        b, l, d = x.size()

        qkv = self.attn_lin1(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)
        q = q.view(b, l, self.attn_head_count, d // self.attn_head_count).transpose(1, 2)
        k = k.view(b, l, self.attn_head_count, d // self.attn_head_count).transpose(1, 2)
        v = v.view(b, l, self.attn_head_count, d // self.attn_head_count).transpose(1, 2)                        


        if self.flash_attn:
            out = torch.nn.functional.scaled_dot_product_attention(
                                                            q, k, v, \
                                                            attn_mask = None, \
                                                            dropout_p=self.attn_dropout_factor if self.training else 0, \
                                                            is_causal=True \
                                                            )

        else:
            out = q @ k.transpose(-1, -2)
            out = out * (1.0 / math.sqrt(k.size(-1)))
            out = out.masked_fill(self.bias[:,:,:l,:l] == 0, float('-inf'))
            out = torch.nn.functional.softmax(out, dim=-1)
            out = self.attn_dropout(out)
            out = out @ v

        out = out.transpose(1, 2).contiguous().view(b, l, d)

        out = self.attn_lin2(out)
        out = self.resid_dropout(out)

        return out

class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm1 = torch.nn.LayerNorm(config["embedding_dim"],  bias=config["bias"])
        self.causal_self_attn = CausalSelfAttention(config)
        self.layernorm2 = torch.nn.LayerNorm(config["embedding_dim"],  bias=config["bias"])
        self.mlp = MultiLevelPerceptron(config)

    def forward(self, x):
        x = self.causal_self_attn(self.layernorm1(x)) + x
        x = self.mlp(self.layernorm2(x)) + x

        return x


class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # check here to make sure our relevant config stats are valid
        assert config["token_count"] is not None
        assert config["embedding_dim"] is not None
        assert config["block_length"] is not None
        assert config["embed_dropout"] is not None
        assert config["layer_count"] is not None

        self.config = config

        self.transformer = torch.nn.ModuleDict(
            dict(
                tkn_embd = torch.nn.Embedding(config["token_count"], config["embedding_dim"]),
                pos_embd = torch.nn.Embedding(config["block_length"], config["embedding_dim"]),
                dropout = torch.nn.Dropout(config["embed_dropout"]),
                blocks = torch.nn.ModuleList([Block(config) for _ in range(config["layer_count"])]),
                lyr_nrm = torch.nn.LayerNorm(config["embedding_dim"], bias=config["bias"])
            )
        )
        self.lm_head = torch.nn.Linear(config["embedding_dim"], config["token_count"], bias=False)

        # weight tying, another innovation from Mr. Karpathy
        # I don't fully understand it but he cites this paper
        # https://paperswithcode.com/method/weight-tying
        self.transformer.tkn_embd.weight = self.lm_head.weight
        self.apply(self._init_weights)

        # WARNING: this is misleading due to weight tying
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * config["layer_count"]))

        param_count = sum(param.numel() for param in self.transformer.parameters())
        print(f"This instantation of EGP has {param_count} parameters.")

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None: # This seems bugged... may need to refactor if we want a LayerNorm with bias
                torch.nn.init.zeros_(module.bias)

    def forward(self, input_block, trgts=None):
        device = input_block.device
        b, h = input_block.size()
        assert h <= self.config["block_length"], f"Input token list too long: we received {h} and our maximum is {self.config['block_length']}."
        pos_seq = torch.arange(0, h, dtype=torch.long, device = device)

        # transformer forward here
        embds_t = self.transformer.tkn_embd(input_block)
        embds_p = self.transformer.pos_embd(pos_seq)

        embds_final = embds_t + embds_p
        out = self.transformer.dropout(embds_final)
        for block in self.transformer.blocks:
            out = block(out)

        logits = self.transformer.lyr_nrm(out)

        if trgts is not None:
            # I'd like to understand the dimensionality of this step better
            logits = self.lm_head(logits)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), trgts.view(-1), ignore_index=-1)

        else:
            # Note that if we have no targets, we don't need the whole matrix to calculate the loss
            # Just the last position for generating next tokens
            logits = self.lm_head(logits[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, model_config, device_type):
        # This enables weight decay for certain parameters in our optimizer
        # See: https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab

        # Nab every param in our model, then filter out the ones that require grad
        param_dict = {name: param for name,param in self.named_parameters()}
        param_dict = {name: param for name,param in param_dict.items() if param.requires_grad}

        # Stuff all our remaining parameters into two categories based how many dimensions they have
        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        nodecay_params = [param for name, param in param_dict.items() if param.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': model_config["weight_decay"]},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(param.numel() for param in decay_params)
        num_nodecay_params = sum(param.numel() for param in nodecay_params)
        print(f"We enable decay for {len(decay_params)} tensors.")
        print(f"These tensors with decay have {num_decay_params} parameters.")
        print(f"We disable decay for {len(nodecay_params)} tensors.")
        print(f"These tensors without decay have {num_nodecay_params} parameters.")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=model_config["learning_rate"],
                                      betas=model_config["betas"],
                                      **extra_args)
        print(f"Is our AdamW fused? {use_fused}")

        return optimizer

