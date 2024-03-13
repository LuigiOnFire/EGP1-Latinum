from torch import nn

class GPT(nn.Module):
    def __init__(self, config):
        super().__init()
        # check here to make sure our relevant config stats are valid
        assert config.token_count is not None
        assert config.embedding_dim is not None
        assert config.block_length is not None
        assert config.dropout is not None
        assert config.block_count is not None
        

        self.transformer = nn.ModuleDict(
            dict(
                tkn_embd = nn.Embedding(config.token_count, config.embedding_dim),
                pos_embd = nn.Embedding(config.block_length, config.embedding_dim),
                drpout = nn.Dropout(config.dropout),
                blocks = [Block for _ in range(config.block_count),
                lyr_nrm = LayerNorm(config.embedding_dim, config,bias)
            )
        )
        self.lm_head = LMHead(config.embedding_dim, config.token_count, bias=False)

    def forward(self, input_block, trgt_tkn):
        device = input_block.device()
        b, h = input_block.dims()
        # do a check here to make sure these params aren't too big for our model
        pos_seq = torch.arrange(0, h)

        # transformer forward here
        embdgs_t = self.transformer.tkn_embd(input_block)
        embdgs_p = self.transformer.pos_embd(pos_seq)
        # do dropout (on what?)
        for block in self.transfoerm.blocks:
            out = block(out)
        out = self.tranformer.lyr_nrm(out)

        logit = self.lm_head(out)
        
        loss = nn.CrossEntropyLoss_(logit, trgt)



