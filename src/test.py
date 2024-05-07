import argparse
from contextlib import nullcontext
import datetime
import json
from pathlib import Path
import torch



import data_load
import data_prep
import model
import tokenizer_spm
import train
from utils import PAD_INDEX
import utils

def load_sample_sents():
    script_dir = Path(__file__).resolve().parent
    file_in = script_dir / "test_sents.txt"
    with open(file_in, 'r', encoding='utf-8') as fi:
        sents_in = fi.readlines()
    
    return sents_in

def encode_sent_list(sents_in, tokenizer_dir):
    sents_in_enc = [] 
    for sent_in in sents_in:
        encoded_str = tokenizer_spm.encode_string(sent_in, tokenizer_dir)
        sents_in_enc.append(encoded_str)

    return sents_in_enc

def pad_trim_input(sents_in_enc, config_model):
    for ind, sent_in in enumerate(sents_in_enc):
        if len(sent_in) > config_model["block_length"]:
            sents_in_enc[ind] = sent_in[-config_model["block_length"]:]
        if len(sent_in) < config_model["block_length"]:
            pads = [PAD_INDEX] * (config_model["block_length"] - len(sent_in))
            sents_in_enc[ind] = pads + sent_in

    assert len(sents_in_enc[ind]) == config_model['block_length'], f"Found sentence of invalid length. \
            Expected a length of {config_model['block_length']} and got {len(sents_in_enc[ind])}"
    
    return sents_in_enc

def add_token_to_sents(sents_tensor, config_model, model, temperature, rnd_sampling, top_k):
    # we'll need to reclip as we keep appending
    temp_sents = sents_tensor if sents_tensor.size(1) <= config_model["block_length"] \
        else sents_tensor[:, -config_model["block_length"]:]

    logits, _ = model(temp_sents.cuda())
    logits = logits[:, -1, :]
    logits /= temperature

    if top_k is not None:
        # topk returns values and indices, we only need the
        # we mask out anything less than the lowes value [values[:, [-1]]]
        values, _ = torch.topk(logits, top_k)
        logits[logits < values[:, [-1]]] = -float('Inf')

    probs = torch.nn.functional.softmax(logits, dim=-1)    

    if rnd_sampling:
        next_token = torch.multinomial(probs, num_samples=1)

    else:
        _, next_token = torch.topk(probs, k=1, dim=-1)

    new_sents_tensor = torch.cat((temp_sents, next_token), dim=1)

    return new_sents_tensor


def gen_test(model, num_new_tokens, config_model, tokenizer_dir, temperature=1.0, rnd_sampling=False, top_k=None):
    """
    Takes sentences from our input file "test_sents.txt", pads them, extends them,
    and writes them to a file. Referenced from nanoGPT's model.generate function.
    """
    # set the model in eval mode
    model.eval()

    # load all the sentences from the file
    sents_in = load_sample_sents()

    # encode input sentences
    sents_in_enc = encode_sent_list(sents_in, tokenizer_dir)

    # reshape/pad as needed as arrays
    sents_in_enc = pad_trim_input(sents_in_enc, config_model)

    # convert all to tensor
    sents_tensor = torch.LongTensor(sents_in_enc)
    sents_tensor = sents_tensor.to(config_model["device"])

    # load them all into model and make new tokens
    for _ in range(num_new_tokens):
        sents_tensor = add_token_to_sents(sents_tensor, config_model, model, temperature, rnd_sampling, top_k)

    # decode the sentences and organize into list
    decoded_tokens = [tokenizer_spm.decode_string(row.tolist(), tokenizer_dir) for row in sents_tensor]
    print(decoded_tokens)
    finished_sents = '\n'.join(decoded_tokens)

    # make output file name
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_dir", help="Directory containing model data", nargs='?')
    parser.add_argument("-mt", "--model_type", dest="model_type", help="Grab settings for new model sizing", nargs='?')
    parser.add_argument("-n", "--name", dest="model_name", help="Get an optional name for the new model", nargs='?')

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    # Set up all model params
    config_model = utils.setup_model_config()
    print("Model params set!")
    config_model = utils.use_preset_config_model(config_model, args.model_type)

    config_model["device"] = device

    model =  model.GPT(config_model)
    print("Model prepared!")
    
    model_dir = utils.all_models_dir / args.model_dir
    loaded_state_dict = None
    data_json = None
    log_text = None
    for f in model_dir.iterdir():
        if f.suffix == ".pth":
            loaded_state_dict = f

        elif f.suffix == ".json":
            data_json = f                
        
        elif f.suffix == ".txt":
            log_txt = f
    
    print(loaded_state_dict)
    model.load_state_dict(torch.load(loaded_state_dict))

    # load model data json
    with open(data_json, 'r', encoding="utf8") as file:
        model_data = json.load(file)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    num_new_tokens = 128 # for when we test

        
    model.to(device)
    # load the model 
    gen_test(model, num_new_tokens, config_model, temperature=0.5, rnd_sampling=True, top_k=1)

def test(model, model_dir, config_model, config_train, test_batches=400, ctx=nullcontext()):
    # check if our test data exists and if not tokenize it
    # TODO: come up with a better system to choose between our 4 datasets
    # data_prep.prep_test_data(model_dir)

    test_data_dir = utils.test_data_dir

    dataset = data_load.TokenizedLatinDataset(test_data_dir,
        config_model["block_length"])
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config_train["batch_size"], 
        shuffle=True,
        pin_memory = True, # not sure what this is
        num_workers = config_train["num_workers"]
    )

    device = config_model["device"]

    print("Estimating loss on test set...")
    model.eval()
    test_loss_all = []
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # TODO: do we need "with ctx:" here? probably...
            with ctx:                
                logits, test_loss = model(x, y)

            test_loss = test_loss.item()
            test_loss_all.append(test_loss)

            if batch % 100 == 0:
                train.show_training_data(x, y, logits, model_dir, num=1)
                avg_loss = sum(test_loss_all) / len(test_loss_all)
            # would take too long to iterate through everything
            if batch > test_batches:
                break

    avg_loss = sum(test_loss_all) / len(test_loss_all)
    print(f"Test Error: Avg loss: {avg_loss:>8f}\n")

    return avg_loss

if __name__ == "__main__":
    test()
