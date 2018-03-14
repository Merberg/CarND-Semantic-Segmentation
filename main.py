import os.path
import tensorflow as tf
import helper
import warnings
import time
from moviepy.editor import VideoFileClip
import numpy as np
import scipy.misc
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

LOGDIR = "/tmp/vgg_seg/"

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_pool3_tensor_name = 'layer3_out:0'
    vgg_pool4_tensor_name = 'layer4_out:0'
    vgg_pool7_tensor_name = 'layer7_out:0'
    
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_pool3 = graph.get_tensor_by_name(vgg_pool3_tensor_name)
    vgg_pool4 = graph.get_tensor_by_name(vgg_pool4_tensor_name)
    vgg_pool7 = graph.get_tensor_by_name(vgg_pool7_tensor_name)
    
    return input_image, keep_prob, vgg_pool3, vgg_pool4, vgg_pool7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_pool3, vgg_pool4, vgg_pool7, num_classes):
    """
    Create the layers for a fully convolutional network.  
    Build skip-layers using the vgg layers.
    :param vgg_pool3: TF Tensor for VGG Layer 3 output
    :param vgg_pool4: TF Tensor for VGG Layer 4 output
    :param vgg_pool7: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    REG_SCALE = 1e-3
    l7_conv2d = tf.layers.conv2d(vgg_pool7, num_classes, 1, padding='same', 
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(REG_SCALE), 
                                 name="layer7_conv1x1")
    output = tf.layers.conv2d_transpose(l7_conv2d, num_classes, 4, strides=(2,2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(REG_SCALE),
                                        name="layer7_convT")
        
    l4_conv2d = tf.layers.conv2d(vgg_pool4, num_classes, 1, padding='same',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(REG_SCALE),
                                 name="layer4_conv1x1")
    l4_add = tf.add(output, l4_conv2d, name="layer4_add")
    output = tf.layers.conv2d_transpose(l4_add, num_classes, 4, strides=(2,2), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(REG_SCALE),
                                        name="layer4_convT")
        
    l3_conv2d = tf.layers.conv2d(vgg_pool3, num_classes, 1, padding='same',
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(REG_SCALE),
                                 name="layer3_conv1x1")
    l3_add = tf.add(output, l3_conv2d, name="layer3_add")
    output = tf.layers.conv2d_transpose(l3_add, num_classes, 16, strides=(8,8), padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(REG_SCALE),
                                        name="layer3_convT")
        
    return output

tests.test_layers(layers)


def optimize(nn_last_layer, truth_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param truth_label: TF Placeholder for the ground truth label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))    
    
    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=truth_label, logits=logits),
                              name="xent")
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        xent = xent + sum(reg_losses)
    
    with tf.name_scope("train"):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(xent)
    
    return logits, train_op, xent

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             truth_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data. 
            Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param truth_label: TF Placeholder for ground truth label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    LEARNING_RATE = 1e-4
    KEEP_PROB = 0.5
    
    # Using TensorBoard to generate visuals
    tf.summary.scalar('loss', cross_entropy_loss)
    tf.summary.histogram('histogram loss', cross_entropy_loss)
    summary_out = tf.summary.merge_all()
    
    sess.run(tf.global_variables_initializer())

    writer_folder = str(time.time());
    writer = tf.summary.FileWriter(LOGDIR+writer_folder, sess.graph)
    writer.add_graph(sess.graph)
    
    print("Training with {} Batches...".format(batch_size))
    i = 0
    
    for epoch_i in range(epochs):
        print("EPOCH {}".format(epoch_i))
        
        for batch_images, batch_labels in get_batches_fn(batch_size):            
            _, loss, s = sess.run([train_op, cross_entropy_loss, summary_out],
                                  feed_dict={input_image: batch_images, 
                                             truth_label: batch_labels,
                                             keep_prob: KEEP_PROB,
                                             learning_rate: LEARNING_RATE})
            print("\tloss {:.5f}".format(loss))
            writer.add_summary(s, i)
            i += 1
    
    writer.close()
    print("Run `tensorboard --logdir {}` to see the results.".format(LOGDIR+writer_folder))
    pass
    
tests.test_train_nn(train_nn)

def find_lane(img, sess, logits, keep_prob, image_pl, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    image = scipy.misc.imresize(img, image_shape)

    im_softmax = sess.run(
        [tf.nn.softmax(logits)],
        {keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return street_im

def clip_find_lane(clip, sess, logits, keep_prob, image_pl, image_shape):
    def clip_lane_finding(clip_image):
        return find_lane(clip_image, sess, logits, keep_prob, image_pl, image_shape)
    return clip.fl_image(clip_lane_finding)

def run():
    NUM_CLASSES = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    EPOCHS = 30
    BATCH_SIZE = 8

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    
    # TF Placeholders
    truth_label = tf.placeholder(tf.float32, shape=[None, image_shape[0], image_shape[1], NUM_CLASSES], 
                                 name="truth_lbls")
    learning_rate = tf.placeholder(tf.float32)

    with tf.Session() as sess:    
        
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, truth_label, learning_rate, NUM_CLASSES)
        
        # Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, 
                 truth_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        

        # OPTIONAL: Apply the trained model to a video
        # See advanced lane finding from Term 1
        #clip1 = VideoFileClip(data_dir + 'harder_challenge_video.mp4').subclip(1,5)
        #projectClip = clip1.fx(clip_find_lane, sess, logits, keep_prob, input_image, image_shape)
        #projectClip.write_videofile(runs_dir + '/harder_challenge_results.mp4', audio=False)


if __name__ == '__main__':
    run()
