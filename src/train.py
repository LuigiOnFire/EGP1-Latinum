import data_load
import data_prep
import model
import torch
import test

class ConfigNode():
    # Note: if this needs to be extended for any reason in the future
    # just use yacs instead
    pass

def setup_config_train():
    config_train = ConfigNode()
    config_train.token_data_dir = Path(__file__).parent.parent / "data" / "token_data"
    config_train.batch_size = 64 # Taken from miniGPT, seems small?
    config_train.vocab_size = # look at encoder doc, or better yet extract automatically
    training-config.num_workers = 4 # From miniGPT

    return config_train
    
def setup_model_config():
    config_model = ConfigNode()
    config_model.embedding_dim = 192 # From miniGPT's miniGPT, start small maybe bigger later
    config_model.bias = None # bias for the linear layers in the mlp 
    config_model.embed_dropout = 0 # miniGPT says 0 for pretraining, 0.1 for finetuning
    config_model.resid_dropout = 0 # if we have overfitting issues let's revisit this
    config_model.attn_dropout = 0
    config_model.attn_head_count = 6 # from minigpt
    config_model.block_length = 1024 
    config_model.layer_count = 6 #from minigpt
    config_model.token_count = # how do we determine this?
    config_model.learning_rate = 5e-4 # from minigpt 
    config_model.betas = (0.9, 0.95)
    config_model.max_iters = 1000 # IMPLEMENT THIS

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch,  (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f?} [{current:>5d}/{size:>5d}]")   

    datetime = today.strftime("%d-%m-%Y_%H-%M")
    torch.save(model.state_dict(), f"model_{datetime}.pth")
    print("Saved PyTorch Model State to model.pth")

if __name__ == "__main__":
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    
    print(f"Using {device} as the device")

    # Set up all trainer params
    config_train = setup_config_train()

    # Set up all model params
    config_model = setup_model_config()

    # run data_prep if it hasn't already #TODO: make an argument for this
    data_prep.prep_data()
    dataset = data_load.TokenizedLatinDataset(config.token_data_dir, config_train.batch_size)
    batch_sampler = data_load.BatchSampler(dataset, config_train.batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config_train.batch_size, 
        shuffle=False,
        pin_memory = True, # not sure what this is
        num_workers = config.num_workers
    )

    #setup our 

#     model = NeuralNetwork().to(device)
#     print(model)
#     epochs = 5
#     for t in range(epochs):
#         print(f"Epoch {t + 1}\n" + "-" * 30)
#         train(train_dataloader, model, loss_fn, optimizer)
#         # test(test_dataloader, model, loss_fn)
#     print("Done!")
