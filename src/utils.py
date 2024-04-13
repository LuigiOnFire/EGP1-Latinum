# For pad/stuffing Latinitium suggests "fartura" or "tomentum" if we want to be cheeky
PAD_TOKEN = "[PAD]"
PAD_INDEX = 0 # putting this back to zero, if it miraculously works and then breaks again, that's why

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
    # doesn't recursively delete directories
    # doesn't need to
    for f in dirname.iterdir():
        f.unlink()
