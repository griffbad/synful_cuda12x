import json

import numpy as np
import tensorflow as tf

from funlib.learn.tensorflow import models


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
    voxel_size = tuple(parameter['voxel_size']) # only needed for computing
    # field of view. No impact on the actual architecture.

    assert unet_model == 'vanilla' or unet_model == 'dh_unet', \
        'unknown unetmodel {}'.format(unet_model)

    # Create the model using tf.function for TF 2.x compatibility
    @tf.function
    def model(raw_input):
        # b=1, c=1, d, h, w
        raw_batched = tf.reshape(raw_input, (1, 1) + input_shape)

        # b=1, c=fmap_num, d, h, w
        outputs, fov, voxel_size_computed = models.unet(raw_batched,
                                               fmap_num, fmap_inc_factor,
                                               downsample_factors,
                                               num_heads=num_heads,
                                               voxel_size=voxel_size)
        if num_heads == 1:
            outputs = (outputs, outputs)

        # b=1, c=3, d, h, w
        partner_vectors_batched, fov = models.conv_pass(
            outputs[0],
            kernel_sizes=[1],
            num_fmaps=3,
            activation=None,  # Regression
            name='partner_vector')

        # b=1, c=1, d, h, w
        syn_indicator_batched, fov = models.conv_pass(
            outputs[1],
            kernel_sizes=[1],
            num_fmaps=1,
            activation=None,
            name='syn_indicator')

        return partner_vectors_batched, syn_indicator_batched, fov, voxel_size_computed

    # Build a concrete function to get output shapes
    raw_spec = tf.TensorSpec(shape=input_shape, dtype=tf.float32)
    concrete_fn = model.get_concrete_function(raw_spec)
    
    # Create dummy input to get shapes
    dummy_input = tf.zeros(input_shape, dtype=tf.float32)
    partner_vectors_batched, syn_indicator_batched, fov, voxel_size_computed = model(dummy_input)
    
    print('unet has fov in nm: ', fov)
    print('fov in nm: ', fov)

    # d, h, w
    output_shape = tuple(syn_indicator_batched.shape[2:])  # strip batch and channel dimension.
    syn_indicator_shape = output_shape

    # c=3, d, h, w
    partner_vectors_shape = (3,) + syn_indicator_shape

    print("input shape : %s" % (input_shape,))
    print("output shape: %s" % (output_shape,))

    # Create a SavedModel instead of meta graph for TF 2.x
    class SynfulModel(tf.Module):
        def __init__(self):
            super().__init__()
            
        @tf.function(input_signature=[
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='raw'),
            tf.TensorSpec(shape=partner_vectors_shape, dtype=tf.float32, name='gt_partner_vectors'),
            tf.TensorSpec(shape=syn_indicator_shape, dtype=tf.float32, name='gt_syn_indicator'),
            tf.TensorSpec(shape=syn_indicator_shape, dtype=tf.float32, name='indicator_weight'),
            tf.TensorSpec(shape=syn_indicator_shape, dtype=tf.float32, name='vectors_mask'),
            tf.TensorSpec(shape=syn_indicator_shape, dtype=tf.bool, name='gt_mask'),
        ])
        def __call__(self, raw, gt_partner_vectors, gt_syn_indicator, indicator_weight, vectors_mask, gt_mask):
            partner_vectors_batched, syn_indicator_batched, fov, voxel_size_computed = model(raw)
            
            # c=3, d, h, w
            pred_partner_vectors = tf.reshape(partner_vectors_batched, partner_vectors_shape)
            
            # d, h, w
            pred_syn_indicator = tf.reshape(syn_indicator_batched, syn_indicator_shape)
            pred_syn_indicator_out = tf.sigmoid(pred_syn_indicator)
            
            vectors_mask_bool = tf.cast(vectors_mask, tf.bool)

            # Calculate losses using TF 2.x APIs
            partner_vectors_loss_mask = tf.reduce_mean(
                tf.boolean_mask(
                    tf.square(gt_partner_vectors - pred_partner_vectors),
                    tf.reshape(vectors_mask_bool, (1,) + syn_indicator_shape)
                )
            )

            syn_indicator_loss_weighted = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=gt_syn_indicator,
                    logits=pred_syn_indicator
                ) * indicator_weight
            )
            
            loss = m_loss_scale * syn_indicator_loss_weighted + d_loss_scale * partner_vectors_loss_mask
            
            return {
                'pred_partner_vectors': pred_partner_vectors,
                'pred_syn_indicator': pred_syn_indicator,
                'pred_syn_indicator_out': pred_syn_indicator_out,
                'loss': loss,
                'loss_vectors': partner_vectors_loss_mask,
                'loss_indicator': syn_indicator_loss_weighted,
            }

        @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='raw')])
        def predict(self, raw):
            partner_vectors_batched, syn_indicator_batched, fov, voxel_size_computed = model(raw)
            
            pred_partner_vectors = tf.reshape(partner_vectors_batched, partner_vectors_shape)
            pred_syn_indicator = tf.reshape(syn_indicator_batched, syn_indicator_shape)
            pred_syn_indicator_out = tf.sigmoid(pred_syn_indicator)
            
            return {
                'pred_partner_vectors': pred_partner_vectors,
                'pred_syn_indicator_out': pred_syn_indicator_out,
            }

    # Create and save the model
    synful_model = SynfulModel()
    
    # Save using SavedModel format
    model_path = f"{name}_model"
    tf.saved_model.save(synful_model, model_path)

    names = {
        'raw': 'raw:0',
        'gt_partner_vectors': 'gt_partner_vectors:0',
        'pred_partner_vectors': 'pred_partner_vectors:0',
        'gt_syn_indicator': 'gt_syn_indicator:0',
        'pred_syn_indicator': 'pred_syn_indicator:0',
        'pred_syn_indicator_out': 'pred_syn_indicator_out:0',
        'indicator_weight': 'indicator_weight:0',
        'vectors_mask': 'vectors_mask:0',
        'gt_mask': 'gt_mask:0',
        'loss': 'loss:0',
        'optimizer': 'optimizer:0',
        'summary': 'summary:0',
        'input_shape': input_shape,
        'output_shape': output_shape,
        'model_path': model_path
    }

    names['outputs'] = {'pred_syn_indicator_out':
                            {"out_dims": 1, "out_dtype": "uint8"},
                        'pred_partner_vectors': {"out_dims": 3,
                                                 "out_dtype": "float32"}}
    if m_loss_scale == 0:
        names['outputs'].pop('pred_syn_indicator_out')
    if d_loss_scale == 0:
        names['outputs'].pop('pred_partner_vectors')

    with open(name + '_config.json', 'w') as f:
        json.dump(names, f)

    # Count parameters in TF 2.x way
    total_parameters = 0
    for variable in synful_model.trainable_variables:
        # shape is an array of tf.Dimension
        shape = variable.shape
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    print("Number of parameters:", total_parameters)
    print("Estimated size of parameters in GB:",
          float(total_parameters) * 8 / (1024 * 1024 * 1024))


if __name__ == "__main__":
    """
    
    Script to generate a tensorflow network. Needs to be run before training.
    
    This script generates a tensorflow meta file (train_net.meta ) that defines network 
    architecture and training parameters for tensorflow. It also generates a 
    config file for the gunpowder train script (train_net_config.json).
    

    Argument 1: parameter json file path.
    
    Example usage: python generate_network.py parameter.json
    """

    with open('parameter.json') as f:
        parameter = json.load(f)

    mknet(parameter, name='train_net')

    # Bigger network used for large datasets, make it as big as gpu memory allows.
    parameter['input_size'] = (90, 1132, 1132)
    mknet(parameter, name='test_net')
