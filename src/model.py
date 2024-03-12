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
        

        self.transformer = nn.ModuleDict(dict(
        tkn_embd = nn.Embedding(config.token_count, config.embedding_dim),
        pos_embd = nn.Embedding(config.block_length, config.embedding_dim),
        dropout_layer = nn.Dropout(config.dropout)
        for i in range(0, config.block_count): # nope gotta use a list comprehension for this
        )

