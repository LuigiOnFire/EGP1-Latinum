import datetime
import json
from pathlib import Path

from cltk.tokenizers import LatinTokenizationProcess
from cltk.core.data_types import Doc
import torch

import torch.autograd.profiler as profiler
import data_load
import data_prep
import model
import utils
from utils import PAD_INDEX



class ConfigNode():
    # Note: if this needs to be extended for any reason in the future
    # just use yacs instead
    pass

def get_vocab_size(token_data_dir):
    filepath = str(token_data_dir / "encoder.json")
    print(f"Loading vocab from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        vocabsize = len(json.loads(file.read()))
        return vocabsize

def setup_config_train():
    config_train = ConfigNode()
    config_train.token_data_dir = Path(__file__).parent.parent / "token_data"
    config_train.batch_size = 32 # Taken from miniGPT, then needed to reduce
    config_train.num_workers = 8 # From miniGPT
    config_train.epochs = 1

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
        print("Get batch")
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        print("Forward pass")
        with profiler.profile(with_modules=True) as prof:
            _, loss = model(x, y)
        
        # for name, param in model.named_parameters():
        #     print(f"{name} is on {param.device}")

        # Backpropogation
        print("Backwards pass")
        with profiler.record_function("BACKWARDS PASS"):
            loss.backward()
        print("grad step")
        optimizer.step()
        print("Zero grad")
        optimizer.zero_grad(set_to_none=True)

        print(prof.key_averages().table(sort_by="cpu_time_total"))

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")   

    time_now = datetime.datetime.now()
    now_str = time_now.strftime("%d-%m-%Y_%H-%M")
    torch.save(model.state_dict(), f"model_{now_str}.pth")
    print("Saved PyTorch Model State to model.pth")

def test(model, num_new_tokens, config_train, config_model, temperature=1.0, rnd_sampling=False):
    """
    Takes sentences from our input file "test_sents.txt", pads them, extends them,
    and writes them to a file. Referenced from nanoGPT's model.generate function.
    """
    # set the model in eval mode
    model.eval()

    # load all the sentences from the file
    script_dir = Path(__file__).resolve().parent
    file_in = script_dir / "test_sents.txt"
    with open(file_in, 'r', encoding='utf-8') as fi:
        sents_in = fi.readlines()

    # encode input sentences
    # SUGGESTION: make a tokenize_latin_sentence and move to util?
    tokenizer_process = LatinTokenizationProcess()
    sents_in_tok = [] # sentences in tokenized
    for sent_in in sents_in:
        token_sent = tokenizer_process.run(input_doc=Doc(raw=sent_in)).tokens
        sents_in_tok.append(token_sent)

    token_data_dir = config_train.token_data_dir
    enc_filepath = str(token_data_dir / "encoder.json")
    with open(enc_filepath, 'r', encoding='utf-8') as file:
        encoder = json.loads(file.read())

    # sentences in encoded
    # hmm we need to map unknown tokens to [unk] or something
    try:
        sents_in_enc = [[encoder[token] for token in row] for row in sents_in_tok]
    except KeyError:
        pass

    # reshape/pad as needed as arrays
    for sent_in in sents_in_enc:
        if len(sent_in) > config_model.block_length:
            sent_in = sent_in[-config_model.block_length:]
        if len(sent_in) < config_model.block_length:
            pads = [PAD_INDEX] * (config_model.block_length - len(sent_in))
            sent_in = pads + sent_in
            print(sent_in)

        assert len(sent_in) == config_model.block_length, f"Found sentence of invalid length. \
              Expected a length of {config_model.block_length} and got {len(sent_in)}"

    # convert all to tensor
    sents_tensor = torch.LongTensor(sents_in_enc)
    sents_tensor = sents_tensor.to(config_model.device)    

    # load them all into model
    for _ in range(num_new_tokens):
        # we'll need to reclip as we keep appending
        temp_sents = sents_tensor if sents_tensor.size(1) <= config_model.block_length \
            else sents_tensor[:, -config_model.block_length]

        # temp_sents.to(config_model.device)
        logits, _ = model(temp_sents.cuda())

        logits = logits[:, -1, :]
        logits /= temperature

        probs = torch.nn.functional.softmax(logits, dim=-1)

        if rnd_sampling:
            next_token = torch.multinomial(probs, num_samples=1)

        else:
            _, next_token = torch.topk(probs, k=1, dim=-1)

        
        sents_tensor = torch.cat((sents_tensor, next_token), dim=1)

    # write all the out sentences to a dated output file
    dec_filepath = str(token_data_dir / "decoder.json")
    with open(dec_filepath, 'r', encoding='utf-8') as file:        
        decoder = json.loads(file.read())

    decoder = utils.convert_dict_string_string_to_dict_int_string(decoder)    

    decoded_tokens = [[decoder[entry.item()] for entry in row] for row in sents_tensor]

    finished_sents = [' '.join(sent) for sent in decoded_tokens]

    time_now = datetime.datetime.now()
    now_str = time_now.strftime("%d-%m-%Y_%H-%M")

    output_dir = Path(__file__).resolve().parent.parent / "train_output"
    file_out = output_dir / ("output_" + now_str + ".txt")
    
    output_dir.mkdir(exist_ok=True)

    # we're going to assume that spaces don't count as tokens 
    # delete this comment when we confirm this
    with open(file_out, 'w+', encoding='utf-8') as fo:
        fo.writelines(finished_sents)
        

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
        
    print(f"Using {device} as the device")

    # Set up all trainer params
    config_train = setup_config_train()
    print("Training params set!")

    # Set up all model params
    config_model = setup_model_config(config_train.token_data_dir)
    print("Model params set!")
    config_model.device = device
    # DEBUG REMOVE
    torch.cuda.synchronize()

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
    num_new_tokens = 32
    for t in range(epochs):
        print(f"Epoch {t + 1}\n" + "-" * 30)
        train(dataloader, model, optimizer, device)
        #test(model, num_new_tokens, config_train,
        #     config_model, temperature=1.0, rnd_sampling=False)
    print("Done!")
