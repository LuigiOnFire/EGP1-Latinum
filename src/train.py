import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

from cltk.tokenizers import LatinTokenizationProcess
from cltk.core.data_types import Doc
import shutil
import torch

import torch.autograd.profiler as profiler
import data_load
import data_prep
import model
import random
import test
import tokenizer_spm
import utils

# only useful if you want to use the CLTK tokenizer
def get_vocab_size(token_data_dir):
    filepath = str(token_data_dir / "encoder.json")
    print(f"Loading vocab from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        vocabsize = len(json.loads(file.read()))
        return vocabsize

# taken from nanoGPT, seems like a cool idea
def get_decayed_lr(iter, config_train):

    # unpack a bit to make code more legible
    warmup_iters = config_train["warmup_iters"]
    cooldown_iters = config_train["cooldown_iters"]
    lr_sml = config_train["learning_rate_sml"]
    lr_lrg = config_train["learning_rate_lrg"]

    # three cases
    # Still warming up...
    if iter < warmup_iters:
        return lr_lrg * iter / warmup_iters
    # Cosine arc between warmup and decay
    
    if warmup_iters <= iter <= cooldown_iters:
        cooldown_ratio = (iter - warmup_iters) / (cooldown_iters - warmup_iters)
        assert 0 <= cooldown_ratio <= 1

        # the cosine swings from +1 to -1
        # so this coeff swings from 0 to -1
        coeff = 0.5 * (1.0 + math.cos(math.pi * cooldown_ratio))
        return lr_sml + coeff * (lr_lrg - lr_sml)
    
    assert iter > cooldown_iters # this is the only condition under which we should reach here
    return lr_sml

def show_training_data(x, y, logits, tokenizer_dir, num=None):
    if num == None:
        num = len(x)
    last_tokens = 16
    in_decoded = [tokenizer_spm.decode_string(row[-last_tokens:].tolist(), tokenizer_dir) for row in x]

    logits = logits[:, -1, :]    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    _, next_token = torch.topk(probs, k=1, dim=-1)
    guessed_decoded = [tokenizer_spm.decode_string(row.tolist(), tokenizer_dir) for row in next_token]

    ans_encoded = y
    ans_decoded = [tokenizer_spm.decode_string(row.tolist()[-1], tokenizer_dir) for row in ans_encoded]
    
    # only bother with the first one, but you can easily print all of them if you want

    for i in range(num):
        print("The sentence is:")
        print(in_decoded[i])
        print(f"We guessed: {guessed_decoded[i]}")
        print(f"The correct answer was: {ans_decoded[i]}")
        print("\n")

def train(dataloader, model, optimizer, device, model_data, config_train, log_text, model_dir, tokenizer_dir):
    backup_time = timedelta(minutes = 60)

    size = len(dataloader.dataset)

    batch_checkpoint = model_data["batch_checkpoint"]    
    losses = model_data["partial_loss_history"]
    num_new_tokens = 32 # for when we test

    # Not sure you can acces batch_size with dataloader.batch_size, could be a problem here
    iter = batch_checkpoint + 1 # if this starts at 0 it zeros out our lr

    # We made need to convert this later on if this causes problems
    time_started = datetime.strptime(model_data["time_last_trained"], "%d-%m-%Y_%H-%M-%S")

    lr = config_model["learning_rate"]
    old_loss = None
    
    for batch, (x, y) in enumerate(dataloader):
        model.train()

        if config_train["lr_decay"] == True:
            lr = get_decayed_lr(iter, config_train)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        x, y = x.to(device), y.to(device)

        # Compute prediction error
        with ctx:
            logits, loss = model(x, y)

        # Backpropogation
        loss.backward()
        if config_train["grad_clip"] is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config_train["grad_clip"])

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            losses.append(loss)
            loss_status = f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]"
            lr_status = f"Using an lr of {lr}"

            print(loss_status)
            print(lr_status)            
            show_training_data(x, y, logits, tokenizer_dir, 1) # can be factored out to happen more often for fun!

            log_text += f"\n{loss_status}"
            log_text += f"\n{lr_status}"

            time_elapsed = datetime.now() - time_started

            if time_elapsed >= backup_time:

                # "+ 1" so that checkpoint can mean "the first batch we do"
                # and hence have a default of 0
                test_loss = test.test(model, tokenizer_dir, config_model, config_train, 100, ctx)
                model_data["loss_history_test"].append(test_loss)
                model_data["batch_checkpoint"] = batch + 1
                model_data["partial_loss_history"] = losses
                model_dir = save_model(model, model_data, all_models_dir, log_text, model_dir)
                # Trying to see if topk=5 gives us better results
                test.gen_test(model, num_new_tokens,
                    config_model, tokenizer_dir, temperature=1.0, rnd_sampling=True, top_k=8)                
                
                # reset the time we measure from
                time_started = datetime.now()
        # if old_loss and old_loss < 7 and loss > 7 and loss - old_loss > 0.75:
        #     print("Possible collapse detected!")
        #     show_training_data(x, y, logits, tokenizer_dir)

        old_loss = loss
        
        iter += 1

    model_data["batch_checkpoint"] = 0
    avg_loss = sum(losses) / len(losses)
    print(f"Average loss: {avg_loss}")
    model_data["loss_history"].append(avg_loss)
    model_data["partial_loss_history"] = []
    model_data["epochs_trained"] += 1
    model_dir = save_model(model, model_data, all_models_dir, log_text, model_dir)
    test.gen_test(model, num_new_tokens,
        config_model, tokenizer_dir, temperature=1.0, rnd_sampling=True, top_k=8)

def update_time_last_trained(model_data):
    time_now = datetime.now()
    now_str_long = datetime.strftime(time_now, "%d-%m-%Y_%H-%M-%S")
    model_data["time_last_trained"] = now_str_long

def init_new_model_data(work_names, config_model, model_name):
    # other data we need
    # model_id
    # workcount + date + trainstart
    time_now = datetime.now()
    # strftiem makes a string
    now_str_short = datetime.strftime(time_now, "%d%m%Y%H%M%S")
    now_str_long = datetime.strftime(time_now, "%d-%m-%Y_%H-%M-%S")

    # optional name for the model
    # hoping args knows the default to None
    model_data["model_name"] = model_name
        
    work_count = len(work_names)
    serial_no = f"{work_count}_{now_str_short}_{now_str_short}"

    if model_data["model_name"] is not None:
        serial_no = f"{model_data['model_name']}_{serial_no}"
    
    model_data["serial_no"] = serial_no

    # epochs trained before
    model_data["epochs_trained"] = 0

    # batches done in last epoch
    model_data["batch_checkpoint"] = 0

    # time model started
    # let's be consistent and make these always strings
    model_data["time_first_trained"] = now_str_long

    # time last trained
    # let's be consistent and make these always long strings
    model_data["time_last_trained"] = now_str_long

    # total time trained    
    # time_trained will always be in seconds
    model_data["time_trained"] = 0

    # on which texts trained
    model_data["works_trained_on"] = work_names

    # all model params
    model_data["config_model"] = config_model

    # losses over each epoch
    model_data["loss_history"] = []

    # losses in training data
    model_data["loss_history_test"] = []

    # loss history within a particular epoch, in case we need to load mid epoch
    model_data["partial_loss_history"] = []

    # random seed for data loader
    model_data["data_seed"] = random.randint(0, 2048)    

    return model_data

def save_model(model, model_data, all_models_dir, log_text, old_model_dir):
    # The short form is for the serial number
    # The long form is for the standaloen fields
    
    str_format_long = "%d-%m-%Y_%H-%M-%S"
    str_format_short = "%d%m%Y%H%M%S"

    time_now = datetime.now()
    
    now_str_short = time_now.strftime(str_format_short)
    now_str_long = time_now.strftime(str_format_long)

    # update time trained 
    # whenever we assign this to model_data, it should be in raw seconds
    time_last_trained = datetime.strptime(model_data["time_last_trained"], "%d-%m-%Y_%H-%M-%S")
    prev_time_trained = float(model_data["time_trained"])
    new_time_trained = (datetime.now() - time_last_trained).total_seconds()
    total_time_trained = (prev_time_trained + new_time_trained)
    model_data["time_trained"] = total_time_trained

    # make building serial no its own function
    model_data["time_last_trained"] = now_str_long    

    # should put this in the model directory too ideally
    serial_no_old = model_data["serial_no"]
    serial_no_split = serial_no_old.split("_")
    serial_no_split[-1] = now_str_short
    serial_no = "_".join(serial_no_split)    
    model_data["serial_no"] = serial_no

    model_dir = all_models_dir / serial_no
    Path.mkdir(model_dir, exist_ok=True)

    pth_path = model_dir / (serial_no + ".pth")
    json_path = model_dir / (serial_no + ".json")
    txt_path = model_dir / (serial_no + ".txt")

    torch.save(model.state_dict(), pth_path)
    print("Saved PyTorch Model State to model.pth") 

    with open(json_path, 'w') as f:
        json.dump(model_data, f, default=str, indent=4)

    with open(txt_path, 'w') as f:
        f.write(log_text)    

    # if our model contains it's own tokenizer
    if (old_model_dir / "m.vocab").exists():
        shutil.copy(old_model_dir / "m.vocab", model_dir / "m.vocab") 
        shutil.copy(old_model_dir / "m.model", model_dir / "m.model") 

    return model_dir    

def initialize_randomize_dataloader(dataset, model_data, random_loader):
    log_index = ""
    if random_loader:
        data_seed = model_data["data_seed"]
        torch.manual_seed(data_seed)
        shuffled_indices = torch.randperm(len(dataset))
        batch_checkpoint = model_data["batch_checkpoint"]    
        
        shuffled_indices = shuffled_indices[batch_checkpoint:]


        # THIS IS FOR DEBUG
        log_index = f"The random indices are {shuffled_indices}\n"

        dataset = torch.utils.data.Subset(dataset, shuffled_indices) 

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config_train["batch_size"], 
        shuffle=False,
        pin_memory = True,
        num_workers = config_train["num_workers"]
    )

    return dataloader, log_index

if __name__ == "__main__":
    random_loader = True
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

    print(f"Using {device} as the device")

    # Set up all trainer params

    config_train = utils.setup_config_train()
    print("Training params set!")    

    # Set up all model params
    config_model = utils.setup_model_config()
    print("Model params set!")
    config_model = utils.use_preset_config_model(config_model, args.model_type)

    config_model["device"] = device

    model =  model.GPT(config_model)
    print("Model prepared!")    

    # dictionary to save to json to save data about the model
    model_data = {}
    all_models_dir = utils.all_models_dir
    Path.mkdir(all_models_dir, exist_ok=True)   

    # if new model
    if args.model_dir == None:
         # run data_prep if it hasn't already #TODO: make an argument for this
        print("Preparing data...")
        _, work_names = data_prep.current_works()
        print("Data prepared!")

        model_data = init_new_model_data(work_names, config_model, args.model_name)
        log_text = ""
        model_dir = all_models_dir / model_data["serial_no"]
        Path.mkdir(model_dir, exist_ok=False)
        # data_prep.prep_data(model_dir)

    else:
    # if model directory specified
    # Load the state dictionary into your model
        model_dir = all_models_dir / args.model_dir
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
            update_time_last_trained(model_data)
            if args.model_name: # this lets us rename the model if we want
                model_data["model_name"] = args.model_name
        
        with open(log_txt, 'r', encoding="utf8") as file:
            # the distinction between log_txt and log_text is kind of annoying but it's workable
            log_text = file.read()
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    torch_dtype_lookup = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }
    ptdtype = torch_dtype_lookup[config_train["dtype"]]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        
    model.to(device)

    print("Configuring optimizer...")
    optimizer = model.configure_optimizers(config_model, device_type)

    tokenizer_dir = Path.home() / "latin-literature-dataset-170M" / "tokenizer_model"

    dataset = data_load.TokenizedLatinDataset(
        config_train["token_data_dir"], config_model["block_length"])    

    # make a random seed for the dataloader that can be preserved across runs
    dataloader, index_log = initialize_randomize_dataloader(dataset, model_data, random_loader)
    log_text += index_log

    print(f"Tokensize was {config_model['token_count']}")

    torch.cuda.empty_cache()

    epochs = config_train["epochs"]
    prev_epochs = model_data["epochs_trained"]
    for t in range(prev_epochs, epochs):
        epoch_message = f"Epoch {t + 1}\n" + "-" * 30 +"\n"
        log_text += epoch_message
        print(epoch_message)
        train(dataloader, model, optimizer, device, model_data, config_train, log_text, model_dir, tokenizer_dir)
        # recreate dataloder 
        dataloader, log_txt = initialize_randomize_dataloader(dataset, model_data, random_loader)
        
    print("Done!")
