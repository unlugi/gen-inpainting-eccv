import tensorflow as tf

from neuralgym.ops.summary_ops import scalar_summary, images_summary

def L1_distance(real,fake):
    return tf.abs(tf.reduce_mean(tf.subtract(real,fake),reduction_indices=[1,2,3]))

def grad_D(costD, d_vars):
    grad_d = tf.gradients(costD, d_vars)
    grad_sum = [tf.nn.l2_loss(grad) for grad in grad_d]
    return tf.reduce_mean(grad_sum)

def gan_lsgan_loss(pos, neg, real, fake, config, name="gan_lsgan_loss"):

    with tf.variable_scope(name):
        gamma = config.GLS_GAMMA

        # Generator loss
        g_loss = tf.reduce_mean(neg)

        # L1 distance between real and fake images
        dist = L1_distance(real, fake)

        # Discriminator loss
        d_loss = tf.reduce_mean( tf.nn.relu( tf.add( tf.subtract(pos, neg), tf.scalar_mul(gamma, dist))))

        # Gradient penalty
        d_loss_penalized = d_loss #+ grad_D(d_loss, real)

        # For tensorboard
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
        scalar_summary('d_loss_penalized', d_loss_penalized)
        scalar_summary('pos_value_avg', tf.reduce_mean(pos))
        scalar_summary('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss_penalized

def gan_glsgan_loss(pos, neg, real, fake, config, name="gan_glsgan_loss"):

    with tf.variable_scope(name):
        gamma = config.GLS_GAMMA
        slope = config.GLS_SLOPE

        # Generator loss
        g_loss = tf.reduce_mean(neg)

        # L1 distance between real and fake images
        dist = L1_distance(real, fake)

        # Discriminator loss
        d_loss = tf.reduce_mean( tf.nn.leaky_relu( tf.add( tf.subtract(pos, neg), tf.scalar_mul(gamma, dist)), alpha=slope ))

        # Gradient penalty
        d_loss_penalized = d_loss #+ grad_D(d_loss, real)

        # For tensorboard
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
        scalar_summary('d_loss_penalized', d_loss_penalized)
        scalar_summary('pos_value_avg', tf.reduce_mean(pos))
        scalar_summary('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss_penalized








