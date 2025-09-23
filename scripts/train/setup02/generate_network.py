import json
import torch

from funlib.learn.torch import models


def mknet(parameter, name):
    learning_rate = parameter['learning_rate']
    input_shape = tuple(parameter['input_size'])
    fmap_inc_factor = parameter['fmap_inc_factor']
    downsample_factors = parameter['downsample_factors']
    fmap_num = parameter['fmap_num']
    unet_model = parameter['unet_model']
    num_heads = 2 if unet_model == 'dh_unet' else 1
    m_loss_scale = parameter['m_loss_scale']
    d_loss_scale = parameter['d_loss_scale']
    voxel_size = tuple(parameter['voxel_size'])

    assert unet_model == 'vanilla' or unet_model == 'dh_unet', \
        'unknown unetmodel {}'.format(unet_model)

    # Create U-Net model
    model = models.UNet(
        in_channels=1,
        num_fmaps=fmap_num,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        num_heads=num_heads
    )

    # Add final conv layers for outputs
    partner_vectors_head = torch.nn.Conv3d(
        in_channels=fmap_num * (fmap_inc_factor ** (len(downsample_factors) - 1)),
        out_channels=3,
        kernel_size=1
    )
    
    syn_indicator_head = torch.nn.Conv3d(
        in_channels=fmap_num * (fmap_inc_factor ** (len(downsample_factors) - 1)),
        out_channels=1,
        kernel_size=1
    )

    # Combine into a complete model
    class SynfulModel(torch.nn.Module):
        def __init__(self, unet, partner_head, indicator_head):
            super().__init__()
            self.unet = unet
            self.partner_head = partner_head
            self.indicator_head = indicator_head
            
        def forward(self, x):
            features = self.unet(x)
            if isinstance(features, tuple):
                partner_features, indicator_features = features
            else:
                partner_features = indicator_features = features
                
            partner_vectors = self.partner_head(partner_features)
            syn_indicator = self.indicator_head(indicator_features)
            
            return partner_vectors, syn_indicator

    complete_model = SynfulModel(model, partner_vectors_head, syn_indicator_head)

    # Calculate output shape by running a dummy forward pass
    with torch.no_grad():
        dummy_input = torch.zeros(1, 1, *input_shape)
        partner_out, indicator_out = complete_model(dummy_input)
        output_shape = indicator_out.shape[2:]  # Remove batch and channel dims

    print("input shape : %s" % (input_shape,))
    print("output shape: %s" % (output_shape,))

    # Create config for training
    config = {
        'model': 'synful_model',
        'input_shape': input_shape,
        'output_shape': tuple(output_shape),
        'learning_rate': learning_rate,
        'm_loss_scale': m_loss_scale,
        'd_loss_scale': d_loss_scale,
        'loss': 'synful_loss',
        'optimizer': 'Adam'
    }

    config['outputs'] = {'pred_syn_indicator_out':
                        {"out_dims": 1, "out_dtype": "uint8"},
                    'pred_partner_vectors': {"out_dims": 3,
                                             "out_dtype": "float32"}}
    if m_loss_scale == 0:
        config['outputs'].pop('pred_syn_indicator_out')
    if d_loss_scale == 0:
        config['outputs'].pop('pred_partner_vectors')

    with open(name + '_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # Save model architecture
    torch.save(complete_model, f'{name}_model.pth')
    
    # Count parameters
    total_parameters = sum(p.numel() for p in complete_model.parameters())
    print("Number of parameters:", total_parameters)
    print("Estimated size of parameters in GB:",
          float(total_parameters) * 4 / (1024 * 1024 * 1024))  # 4 bytes per float32

    return complete_model


if __name__ == "__main__":
    """
    
    Script to generate a PyTorch network. Needs to be run before training.
    
    This script generates a PyTorch model file (train_net_model.pth) that defines the network 
    architecture and a config file for the gunpowder train script (train_net_config.json).
    

    Argument 1: parameter json file path.
    
    Example usage: python generate_network.py parameter.json
    """

    with open('parameter.json') as f:
        parameter = json.load(f)

    mknet(parameter, name='train_net')

    # Bigger network used for large datasets, make it as big as gpu memory allows.
    parameter['input_size'] = (90, 1132, 1132)
    mknet(parameter, name='test_net')
