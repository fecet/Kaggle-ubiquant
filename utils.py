import os


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


def find_gpus(num_of_cards_needed=4):
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus")
    # If there is no ~ in the path, return the path unchanged
    with open(os.path.expanduser("~/.tmp_free_gpus"), "r") as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [(idx, int(x.split()[2])) for idx, x in enumerate(frees)]
    idx_freeMemory_pair.sort(reverse=True)  # 0号卡经常有人抢，让最后一张卡在下面的sort中优先
    idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
    usingGPUs = [
        str(idx_memory_pair[0])
        for idx_memory_pair in idx_freeMemory_pair[:num_of_cards_needed]
    ]
    usingGPUs = ",".join(usingGPUs)
    print("using GPUs:", end=" ")
    for pair in idx_freeMemory_pair[:num_of_cards_needed]:
        print(f"{pair[0]}号，此前空闲：{pair[1]/1024:.1f}GB")
    return usingGPUs


def get_layers_output(model, input):
    from keract import get_activations

    activations = get_activations(model, input, auto_compile=False)

    return activations
