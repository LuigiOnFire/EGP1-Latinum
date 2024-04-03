import datetime
import json
from pathlib import Path

from cltk.tokenizers import LatinTokenizationProcess
from cltk.core.data_types import Doc

import data_load
import data_prep
import model
import torch


class ConfigNode():
    # Note: if this needs to be extended for any reason in the future
    # just use yacs instead
    pass

def get_vocab_size(token_data_dir):
    filepath = str(token_data_dir / "encoder.json")
    print(f"Loading vocab from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        return len(json.loads(file.read()))

def setup_config_train():
    config_train = ConfigNode()
    config_train.token_data_dir = Path(__file__).parent.parent / "data" / "token_data"
    config_train.batch_size = 32 # Taken from miniGPT, then needed to reduce
    config_train.num_workers = 4 # From miniGPT
    config_train.epochs = 1000

    return config_train
    
def setup_model_config(token_data_dir):
    config_model = ConfigNode()
    config_model.embedding_dim = 192 # From miniGPT's miniGPT, start small maybe bigger later
    config_model.bias = None # bias for the linear layers in the mlp 
    config_model.embed_dropout = 0 # miniGPT says 0 for pretraining, 0.1 for finetuning
    config_model.resid_dropout = 0 # if we have overfitting issues let's revisit this
    config_model.attn_dropout = 0
    config_model.attn_head_count = 6 # from minigpt
    config_model.block_length = 512 # reduced from minigpt 
    config_model.layer_count = 6 #from minigpt
    config_model.token_count = get_vocab_size(token_data_dir)
    config_model.learning_rate = 5e-4 # from minigpt 
    config_model.betas = (0.9, 0.95)
    config_model.weight_decay = 0.1 # from minigpt

    return config_model

def train(dataloader, model, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch,  (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = model.loss(pred, y)

        # Backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")   

    time_now = datetime.datetime.now()
    now_str = time_now.strftime("%d-%m-%Y_%H-%M")
    torch.save(model.state_dict(), f"model_{datetime}{now_str}.pth")
    print("Saved PyTorch Model State to model.pth")

def test(model, new_tokens):
    # set the model in eval mode
    model.eval()

    # load all the sentences from the file
    script_dir = Path(__file__).resolve().parent
    file_in = script_dir / "test_sents.txt"
    with open(file_in, 'r', encoding='utf-8') as fi:
        sents_in = fi.readlines()

    # encode input sentences
    # SUGGESTION: make a tokenize_latin_sentence and move to util?
    sents_in_enc = []
    for sent_in in sents_in:
        tokenizer_process = LatinTokenizationProcess()
        tokenized_doc = tokenizer_process.run(input_doc=Doc(raw=sent_in))
        return tokenized_doc.tokens

    # reshape/pad as needed
    for idx, sent_in in enumerate(sents_in):
        sents_in[idx] = sent_in if sent_in.size(1) <= config_model.block_length else 

    # load them all into model
    sents_out = []
    for sent_in in sents_in:
        sents_out.append(model(sent_in))

    # write all the out sentences to a dated output file

if __name__ == "__main__":
    print("Preparing data...")
    data_prep.prep_data()
    print("Data prepared!")

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    
    torch.device = device
    print(f"Using {device} as the device")

    # Set up all trainer params
    config_train = setup_config_train()
    print("Training params set!")

    # Set up all model params
    config_model = setup_model_config(config_train.token_data_dir)
    print("Model params set!")
    config_model.device = device

    # run data_prep if it hasn't already #TODO: make an argument for this
    
    dataset = data_load.TokenizedLatinDataset(
        config_train.token_data_dir, config_model.block_length)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config_train.batch_size, 
        shuffle=False,
        pin_memory = True, # not sure what this is
        num_workers = config_train.num_workers
    )    

    print("Initializing GPT model...")
    model =  model.GPT(config_model)
    print("Model prepared!")

    model.to(device)

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    print("Configuring optimizer...")
    optimizer = model.configure_optimizers(config_model, device_type)

    print(f"Tokensize was {config_model.token_count}")

    torch.cuda.empty_cache()

    epochs = config_train.epochs
    for t in range(epochs):
        print(f"Epoch {t + 1}\n" + "-" * 30)
        train(dataloader, model, optimizer, device)
        # test(test_dataloader, model, loss_fn)
    print("Done!")
