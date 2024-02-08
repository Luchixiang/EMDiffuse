from collections import OrderedDict
import core.util as Util
model_path = 'experiments/train_EMDiffuse-n-large_240125_221819/5180_Network_ema.pth'
import torch
weight_state_dict = dict(torch.load(model_path, map_location=lambda storage, loc: Util.set_device(storage)))



net_weight_state_dict = OrderedDict()
for key in weight_state_dict.keys():
    if 'input_block' in key or 'output_block' in key:

        key_split = key.split('.')
        key_split.insert(4, '0')
        key_replace = '.'.join(key_split)
    elif 'middle_block' in key:
        key_split = key.split('.')
        key_split.insert(3, '0')
        key_replace = '.'.join(key_split)
    else:
        key_replace = key
    if 'in_layers.0' in key:
        net_weight_state_dict[key_replace.replace('in_layers.0', 'in_layers0.0')] = weight_state_dict[key]
    elif 'in_layers.1' in key:
        net_weight_state_dict[key_replace.replace('in_layers.1', 'in_layers0.1')] = weight_state_dict[key]
    elif 'in_layers.2' in key:
        net_weight_state_dict[key_replace.replace('in_layers.2', 'in_layers1.0')] = weight_state_dict[key]
    elif 'out_layers.0' in key:
        net_weight_state_dict[key_replace.replace('out_layers.0', 'out_layers0.0')] = weight_state_dict[key]
    elif 'out_layers.1' in key:

        net_weight_state_dict[key_replace.replace('out_layers.1', 'out_layers1.0')] = weight_state_dict[key]
    elif 'out_layers.2' in key:
        net_weight_state_dict[key_replace.replace('out_layers.2', 'out_layers1.1')] = weight_state_dict[key]
    elif 'out_layers.3' in key:
        net_weight_state_dict[key_replace.replace('out_layers.3', 'out_layers1.2')] = weight_state_dict[key]
    else:
        net_weight_state_dict[key_replace] = weight_state_dict[key]
torch.save(net_weight_state_dict, 'experiments/train_EMDiffuse-n-large_240125_221819/5180_jit_Network_ema.pth')