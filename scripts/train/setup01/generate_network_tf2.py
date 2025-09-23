import json
import os

import numpy as np
import tensorflow as tf

# Enable compatibility with TensorFlow 2.x while maintaining TF 1.x behavior for gradual migration
tf.compat.v1.disable_eager_execution()

from funlib.learn.tensorflow import models


def mknet(parameter, name):
    """
    Create a U-Net for synaptic partner detection compatible with TensorFlow 2.x and CUDA 12.x.
    Optimized for NVIDIA 5090 GPUs with mixed precision support.
    """
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

    assert unet_model in ['vanilla', 'dh_unet'], f'unknown unetmodel {unet_model}'

    # Reset default graph for TF 2.x compatibility
    tf.compat.v1.reset_default_graph()

    # Enable mixed precision for NVIDIA 5090 GPUs
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # Input tensor: d, h, w
    raw = tf.compat.v1.placeholder(tf.float32, shape=input_shape, name='raw_input')

    # Reshape for batch processing: b=1, c=1, d, h, w
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)

    # U-Net with enhanced GPU utilization
    with tf.device('/GPU:0'):  # Ensure GPU placement for NVIDIA 5090
        outputs, fov, voxel_size = models.unet(
            raw_batched,
            fmap_num, 
            fmap_inc_factor,
            downsample_factors,
            num_heads=num_heads,
            voxel_size=voxel_size
        )
        
    if num_heads == 1:
        outputs = (outputs, outputs)
    
    print(f'U-Net has field of view in nm: {fov}')

    # Partner vectors output: b=1, c=3, d, h, w
    partner_vectors_batched, fov = models.conv_pass(
        outputs[0],
        kernel_sizes=[1],
        num_fmaps=3,
        activation=None,  # Regression
        name='partner_vector'
    )

    # Synapse indicator output: b=1, c=1, d, h, w  
    syn_indicator_batched, fov = models.conv_pass(
        outputs[1],
        kernel_sizes=[1],
        num_fmaps=1,
        activation=None,
        name='syn_indicator'
    )
    
    print(f'Output field of view in nm: {fov}')

    # Extract output shape
    output_shape = tuple(syn_indicator_batched.get_shape().as_list()[2:])
    syn_indicator_shape = output_shape
    partner_vectors_shape = (3,) + syn_indicator_shape

    # Reshape outputs
    pred_partner_vectors = tf.reshape(partner_vectors_batched, partner_vectors_shape, name='pred_partner_vectors')
    pred_syn_indicator = tf.reshape(syn_indicator_batched, syn_indicator_shape, name='pred_syn_indicator')

    # Ground truth placeholders
    gt_partner_vectors = tf.compat.v1.placeholder(tf.float32, shape=partner_vectors_shape, name='gt_partner_vectors')
    gt_syn_indicator = tf.compat.v1.placeholder(tf.float32, shape=syn_indicator_shape, name='gt_syn_indicator')
    vectors_mask = tf.compat.v1.placeholder(tf.float32, shape=syn_indicator_shape, name='vectors_mask')
    indicator_weight = tf.compat.v1.placeholder(tf.float32, shape=syn_indicator_shape, name='indicator_weight')
    gt_mask = tf.compat.v1.placeholder(tf.bool, shape=syn_indicator_shape, name='gt_mask')

    # Convert mask to boolean
    vectors_mask_bool = tf.cast(vectors_mask, tf.bool)

    # Loss calculations with improved numerical stability
    with tf.name_scope('losses'):
        # Partner vectors loss (MSE with masking)
        partner_vectors_loss = tf.compat.v1.losses.mean_squared_error(
            gt_partner_vectors,
            pred_partner_vectors,
            weights=tf.reshape(vectors_mask, (1,) + syn_indicator_shape),
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )

        # Synapse indicator loss (weighted sigmoid cross-entropy)
        syn_indicator_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            gt_syn_indicator,
            pred_syn_indicator,
            weights=indicator_weight,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        )
        
        # Combined loss
        total_loss = m_loss_scale * syn_indicator_loss + d_loss_scale * partner_vectors_loss

    # Output predictions
    pred_syn_indicator_out = tf.sigmoid(pred_syn_indicator, name='pred_syn_indicator_out')

    # Training iteration counter
    iteration = tf.Variable(1.0, name='training_iteration', trainable=False)

    # Optimizer with gradient clipping for stability on NVIDIA 5090
    with tf.name_scope('optimization'):
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.95,
            beta2=0.999,
            epsilon=1e-8
        )
        
        # Gradient clipping for numerical stability
        gvs = optimizer.compute_gradients(total_loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs if grad is not None]
        train_op = optimizer.apply_gradients(clipped_gvs, global_step=iteration)

    # TensorBoard summaries
    with tf.name_scope('summaries'):
        tf.compat.v1.summary.scalar('total_loss', total_loss)
        tf.compat.v1.summary.scalar('partner_vectors_loss', partner_vectors_loss)
        tf.compat.v1.summary.scalar('syn_indicator_loss', syn_indicator_loss)
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)
        
        # Add histogram summaries for debugging
        tf.compat.v1.summary.histogram('pred_partner_vectors', pred_partner_vectors)
        tf.compat.v1.summary.histogram('pred_syn_indicator', pred_syn_indicator_out)
        
        summary_op = tf.compat.v1.summary.merge_all()

    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")

    # Export meta graph
    tf.compat.v1.train.export_meta_graph(filename=name + '.meta')

    # Configuration dictionary for gunpowder
    config = {
        'raw': raw.name,
        'gt_partner_vectors': gt_partner_vectors.name,
        'pred_partner_vectors': pred_partner_vectors.name,
        'gt_syn_indicator': gt_syn_indicator.name,
        'pred_syn_indicator': pred_syn_indicator.name,
        'pred_syn_indicator_out': pred_syn_indicator_out.name,
        'indicator_weight': indicator_weight.name,
        'vectors_mask': vectors_mask.name,
        'gt_mask': gt_mask.name,
        'loss': total_loss.name,
        'optimizer': train_op.name,
        'summary': summary_op.name,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'iteration': iteration.name
    }

    # Output specifications
    config['outputs'] = {}
    if m_loss_scale > 0:
        config['outputs']['pred_syn_indicator_out'] = {
            "out_dims": 1, 
            "out_dtype": "uint8"
        }
    if d_loss_scale > 0:
        config['outputs']['pred_partner_vectors'] = {
            "out_dims": 3,
            "out_dtype": "float32"
        }

    # Save configuration
    with open(name + '_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Calculate and display parameter count
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value if dim.value is not None else 1
        total_parameters += variable_parameters
    
    print(f"Number of parameters: {total_parameters:,}")
    print(f"Estimated parameter size in GB: {float(total_parameters) * 4 / (1024**3):.2f}")
    
    return config


if __name__ == "__main__":
    """
    Generate TensorFlow 2.x compatible network for CUDA 12.x and NVIDIA 5090 GPUs.
    
    This script generates a TensorFlow meta file (train_net.meta) that defines the network 
    architecture and training parameters. It also generates a config file for gunpowder
    (train_net_config.json).

    Usage: python generate_network_tf2.py
    """
    
    # Load parameters
    with open('parameter.json') as f:
        parameter = json.load(f)

    print("Generating training network...")
    mknet(parameter, name='train_net')

    # Generate larger test network optimized for NVIDIA 5090 memory capacity
    print("\nGenerating test network for large-scale inference...")
    test_params = parameter.copy()
    test_params['input_size'] = [90, 1132, 1132]  # Larger input for high-memory GPUs
    mknet(test_params, name='test_net')
    
    print("\nNetwork generation completed successfully!")