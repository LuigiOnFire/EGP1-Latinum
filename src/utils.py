from pathlib import Path
import shutil
import torch
# For pad/stuffing Latinitium suggests "fartura" or "tomentum" if we want to be cheeky
# this are no longer useful after using sentencepiece

# this is the default for sentence piece but I'm specifying it anyway 
# PAD_TOKEN should only be used if we're using cltk tokenizer
PAD_TOKEN = "<pad>"
PAD_INDEX = 3

VOCAB_SIZE = 20000

all_models_dir = Path(__file__).parent.parent / "models"

test_data_dir = Path.home() / "latin-literature-dataset-170M" / "bin_test"



def setup_config_train():
    config_train = {}
    config_train["token_data_dir"] = Path.home() / "latin-literature-dataset-170M" / "bin_train"

    config_train["batch_size"] = 8
    config_train["num_workers"] = 12 
    config_train["epochs"] = 12
    config_train["learning_rate_lrg"] = 6e-4
    config_train["learning_rate_sml"] = 0.1 * config_train["learning_rate_lrg"] # taken from nanoGPT;\    

    # doesn't serve any purpose other than calculation here
    # value again from nanoGPT, although should probably be recalculated
    total_iters = 4000000             
    config_train["warmup_iters"] = 2000 
    config_train["cooldown_iters"] = total_iters
    config_train["lr_decay"] = True # bool, idea from nanoGPT
    config_train["dtype"] = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else 'float16'
    config_train["grad_accumulation_steps"] = 40 # nanoGPT uses 5 * 8 = 40, not sure why 5 * 8 is significant
    
    print(f"We ended up using {config_train['dtype']}")
    
    # not implemented yet lol
    config_train["grad_clip"] = 1 # maybe exploding gradient issue so another idea from nanoGPT

    return config_train

def setup_model_config():
    config_model = {}
    config_model["embedding_dim"] = 768 # Same as nanoGPT
    config_model["bias"] = None # bias for the linear layers in the mlp 
    config_model["embed_dropout"] = 0 # miniGPT says 0 for pretraining, 0.1 for finetuning
    config_model["resid_dropout"] = 0 # if we have overfitting issues let's revisit this
    config_model["attn_dropout"] = 0
    config_model["attn_head_count"] = 12 # from nanogpt
    config_model["block_length"] = 512 # reduced from minigpt 
    config_model["layer_count"] = 12 # from nanogpt
    config_model["token_count"] = VOCAB_SIZE
    config_model["learning_rate"] = 3e-5 # gets basically ignored if we use lr_decay
    config_model["betas"] = (0.9, 0.95)
    config_model["weight_decay"] = 0.1 # from minigpt

    return config_model

# note based on Andrej chinchilla calculations, this is compute optimal
# predicted parameters for 3.000000e+06 tokens: 4.762067e+07
# mini has 6.600e6
# basic has 100.1e6
# 11555072

def use_preset_config_model(config_model, model_type):
    # idea borrowed from minGPT
    if model_type == "miniEGP":
        print("Grabbing presets for miniEGP!")
        config_model["layer_count"] = 6
        config_model["attn_head_count"] = 6
        config_model["embedding_dim"] = 192    
        config_model["learning_rate"] = 5e-4
    
    # targeted to be roughly optimal with Andrej's chinchilla calculations
    if model_type == "minusEGP":
        print("Grabbing presets for miniEGP!")
        config_model["layer_count"] = 10
        config_model["attn_head_count"] = 10
        config_model["embedding_dim"] = 480
        config_model["learning_rate"] = 1.3e-4

    if model_type == "basicEGP":
        print("Grabbing presets for basicEGP!")
        config_model["layer_count"] = 12
        config_model["attn_head_count"] = 12
        config_model["embedding_dim"] = 768            
        config_model["learning_rate"] = 3e-5

    return config_model

def convert_dict_string_string_to_dict_int_string(dict_in):
    """
    e.g. used to restore the decoder dictionary when loading
    it from a .json.
    """
    dict_out = {}
    for key, val in dict_in.items():
        try:
            key_int = int(key)

        except ValueError:
            print("While converting dictionry to int, string found key that was not convertible to int.")
            key_int = key

        dict_out[key_int] = val
    return dict_out

def empty_dir(dirname):
    for item in dirname.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            empty_dir(item)


def move_dir(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file():
            dst_path = dst_dir / item.name            
            shutil.move(item, dst_path)
        if item.is_dir():
            sub_dst_dir = dst_dir / item.name
            move_dir(item, sub_dst_dir)

    assert not any(src_dir.iterdir()), "After moving directory, some files were still left over"
    shutil.rmtree(src_dir)
