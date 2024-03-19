"""
Based upon the nanoGPT and miniGPT implementatinos of Andrej Karpathy
"""
from torch import nn

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        

class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # check here to make sure our relevant config stats are valid
        assert config.token_count is not None
        assert config.embedding_dim is not None
        assert config.block_length is not None
        assert config.dropout is not None
        assert config.block_count is not None
        
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                tkn_embd = nn.Embedding(config.token_count, config.embedding_dim),
                pos_embd = nn.Embedding(config.block_length, config.embedding_dim),
                dropout = nn.Dropout(config.dropout),
                blocks = nn.ModuleList([Block(config) for _ in range(config.block_count)),
                lyr_nrm = nn.LayerNorm(config.embedding_dim, bias=config.bias)
            )
        )
        self.lm_head = nn.Linear(config.embedding_dim, config.token_count, bias=False)

    def forward(self, input_block, trgt_tkn):
        device = input_block.device()
        b, h = input_block.dims()
        assert h < self.config.block_size, f"Input token list too long: we received {h} and our maximum is {self.config.block_size."
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
        
        loss = nn.CrossEntropyLoss_(logit, trgt)

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
        optimizer = torch.optim.AdamW(config.optim_groups,
                                      lr=config.learning_rate, 
                                      betas=config.betas, 
                                      **extra_args)
        print("Is our AdamW fused? {use_fused}")

        return optimizer

