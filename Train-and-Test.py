'''
Test Platform: Tensorflow 

Paper Information:

Learnable Descent Algorithm for Nonsmooth Nonconvex Image Reconstruction

Y. Chen, H. Liu, X. Ye, Q. Zhang
'''

import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import tensorflow.contrib.slim as slim
import glob
from time import time
from PIL import Image
import math


Test_Img = './Test_Image'

CS_ratio = 10    # 4, 10, 25, 30,, 40, 50

if CS_ratio == 4:
    n_input = 43
elif CS_ratio == 1:
    n_input = 10
elif CS_ratio == 10:
    n_input = 109
elif CS_ratio == 25:
    n_input = 272
elif CS_ratio == 30:
    n_input = 327
elif CS_ratio == 40:
    n_input = 436
elif CS_ratio == 50:
    n_input = 545


n_output = 1089
batch_size = 64
PhaseNumber_start = 3
PhaseNumber_end = 22
nrtrain = 88912
learning_rate = 0.0001
EpochNum = 100
ddelta = 0.01
channel_number = 32

print('Load Data...')

Phi_data_Name = 'phi_0_%d_1089.mat' % CS_ratio
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi'].transpose()

Training_data_Name = 'Training_Data_Img91.mat'
Training_data = sio.loadmat(Training_data_Name)
Training_inputs = Training_data['inputs']
Training_labels = Training_data['labels']


# Computing Initialization Matrix
XX = Training_labels.transpose()
BB = np.dot(Phi_input.transpose(), XX)
BBB = np.dot(BB, BB.transpose())
CCC = np.dot(XX, BB.transpose())
PhiT_ = np.dot(CCC, np.linalg.inv(BBB))
del XX, BB, BBB, Training_data
PhiInv_input = PhiT_.transpose()
PhiTPhi_input = np.dot(Phi_input, Phi_input.transpose())


Phi = tf.constant(Phi_input, dtype=tf.float32)
PhiTPhi = tf.constant(PhiTPhi_input, dtype=tf.float32)
PhiInv = tf.constant(PhiInv_input, dtype=tf.float32)

X_input = tf.placeholder(tf.float32, [None, n_input])
X_output = tf.placeholder(tf.float32, [None, n_output])

Epoch_i = tf.placeholder(tf.int32)


X0 = tf.matmul(X_input, PhiInv)

PhiTb = tf.matmul(X_input, tf.transpose(Phi))

def imread_CS_py(imgName):
    block_size = 33
    Iorg = np.array(Image.open(imgName), dtype='float32')
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    relative_loss = np.divide(np.linalg.norm(img1 - img2, ord='fro'), np.linalg.norm(img1, ord='fro'))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)), relative_loss

    ######################################################################################################################
def rec(cpkt_model_number, PhaseNumber, regu_num, Prediction):
    #Test
    filepaths = glob.glob(Test_Img + '/*.tif')
    
    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    ERROR_All = np.zeros([1, ImgNum], dtype=np.float32)
    
    
    model_dir = 'Regu_%d_Phase_%d_ratio_0_%d_ISTA_Net_plus_Model-relu-cascade' % (regu_num, PhaseNumber, CS_ratio)
    saver.restore(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, cpkt_model_number))
    
    output_file_name = "PSNR_Results_%s.txt" % (model_dir)
    output_file = open(output_file_name, 'a')
    
    for img_no in range(ImgNum):
    
        imgName = filepaths[img_no]
    
        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(imgName)
        Icol = img2col_py(Ipad, 33).transpose()/255.0
        #print(Ipad.shape)
        Img_input = np.dot(Icol, Phi_input)

        start = time()
        Prediction_value = sess.run(Prediction[-1], feed_dict={X_input: Img_input, Epoch_i: 200})
        end = time()
        print('Pic is %s, time is %.4f' %(imgName, (end - start)))
        print(sess.run(Sign_out, feed_dict={X_input: Img_input, Epoch_i: 200}))
    
        X_rec = col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new)
    
        rec_PSNR, relative_error = psnr(X_rec * 255, Iorg)
    
        img_rec_name = "%s_rec_%s_%d_PSNR_%.2f.png" % (imgName, model_dir, cpkt_model_number, rec_PSNR)
    
        x_im_rec = Image.fromarray(np.clip(X_rec * 255, 0, 255).astype(np.uint8))
        x_im_rec.save(img_rec_name)
        PSNR_All[0, img_no] = rec_PSNR
        ERROR_All[0, img_no] = relative_error
    
    
    output_data = "Avg PSNR is %.2f dB, Avg relative error is %.4f dB,, cpkt NO. is %d, phase number is %d, time is %.4f \n" % (np.mean(PSNR_All), 
                                                                                                                  np.mean(ERROR_All), cpkt_model_number, PhaseNumber, (end - start))

    print(output_data)

    output_file.write(output_data)
    output_file.close()
    return np.mean(PSNR_All), np.mean(ERROR_All)

def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]

def sigma_activation(x_i):
    x_i_delta_sign = tf.sign(tf.nn.relu(tf.abs(x_i) - ddelta))
    x_square = tf.square(x_i)
    x_i_111 = tf.divide(1.0, 4.0 * ddelta)*x_square + 0.5*x_i + 0.25 * ddelta
    x_i_1 = tf.multiply(1.0 - x_i_delta_sign, x_i_111) + tf.multiply(x_i_delta_sign, tf.nn.relu(x_i))
    return x_i_1

def sigma_derivative(x_i):
    x_i_delta_sign = tf.sign(tf.nn.relu(tf.abs(x_i) - ddelta))
    x_i_111 = tf.divide(1.0, 2.0 * ddelta)*x_i + 0.5
    x_i_1 = tf.multiply(1.0 - x_i_delta_sign, x_i_111) + tf.multiply(x_i_delta_sign, tf.sign(tf.nn.relu(x_i)))
    return x_i_1
    
def R(x_i,  Weights0, Weights1, Weights2, Weights3, shrinkage_thresh):
    x_i = tf.reshape(x_i, shape=[-1, 33, 33, 1])
    x_i = tf.nn.conv2d(x_i, Weights0, strides=[1, 1, 1, 1], padding='SAME')
    x_i = sigma_activation(x_i)
    x_i = tf.nn.conv2d(x_i, Weights1, strides=[1, 1, 1, 1], padding='SAME')
    x_i = sigma_activation(x_i)
    #x_i_1 = tf.nn.elu(x_i)
    
    x_i = tf.nn.conv2d(x_i, Weights2, strides=[1, 1, 1, 1], padding='SAME')
    x_i_1 = sigma_activation(x_i)
    

    x_i_1 = tf.nn.conv2d(x_i_1, Weights3, strides=[1, 1, 1, 1], padding='SAME')
    
    
    x_i_1 = tf.norm(x_i_1, ord=2, axis=-1)
    
    greater = tf.sign(tf.nn.relu(x_i_1 - shrinkage_thresh))
    
    less = tf.ones_like(greater) - greater
    
    x_i_1 = tf.multiply(less, tf.divide(tf.square(x_i_1), shrinkage_thresh)) + tf.multiply(greater, x_i_1)
    
    x_i_1 = tf.reshape(x_i_1, shape=[-1, 1089])
    
    x_i_1 = tf.reduce_sum(x_i_1, 1, keep_dims=True)
    
    return x_i_1

def grad_R(x_i, Weights0, Weights1, Weights2, Weights3, shrinkage_thresh):
    x_0 = tf.reshape(x_i, shape=[-1, 33, 33, 1])
    x_shape_0 = tf.shape(x_0)
    x_1 = tf.nn.conv2d(x_0, Weights0, strides=[1, 1, 1, 1], padding='SAME')
    x_shape_1 = tf.shape(x_1)
    x_2 = tf.nn.conv2d(sigma_activation(x_1), Weights1, strides=[1, 1, 1, 1], padding='SAME')
    x_shape_2 = tf.shape(x_2)
    x_3 = tf.nn.conv2d(sigma_activation(x_2), Weights2, strides=[1, 1, 1, 1], padding='SAME')
    x_shape_3 = tf.shape(x_3)
    
    x_i_1 = sigma_activation(x_3)
    x_i_1 = tf.nn.conv2d(x_i_1, Weights3, strides=[1, 1, 1, 1], padding='SAME')
    
    
    norm_along_d = tf.norm(x_i_1, axis=-1)
    
    greater = tf.tile(tf.expand_dims(tf.sign(tf.nn.relu(norm_along_d - shrinkage_thresh)), -1), [1, 1, 1, channel_number])
    
    less = tf.ones_like(greater) - greater
    
    x_i_1_greater = tf.nn.l2_normalize(tf.multiply(greater, x_i_1), dim=-1)
    x_i_1_less = tf.divide(tf.multiply(less, x_i_1), shrinkage_thresh)
    
    x_i_1_out = x_i_1_greater + x_i_1_less
    
    x_i_1_out = tf.nn.conv2d_transpose(x_i_1_out, Weights3, x_shape_3, [1, 1, 1, 1], padding='SAME')
    
    x_i_2 = sigma_derivative(x_3)
    
    x_i_3 = tf.multiply(x_i_1_out, x_i_2)
    
    x_i_3 = tf.nn.conv2d_transpose(x_i_3, Weights2, x_shape_2, [1, 1, 1, 1], padding='SAME')
    x_i_3_1 = sigma_derivative(x_2)
    
    x_i_4 = tf.multiply(x_i_3_1, x_i_3)
    x_i_4 = tf.nn.conv2d_transpose(x_i_4, Weights1, x_shape_1, [1, 1, 1, 1], padding='SAME')
    
    x_i_4_1 = sigma_derivative(x_1)
    
    x_i_5 = tf.multiply(x_i_4_1, x_i_4)
    
    x_i_5 = tf.nn.conv2d_transpose(x_i_5, Weights0, x_shape_0, [1, 1, 1, 1], padding='SAME')
    
    x_i_5 = tf.reshape(x_i_5, shape=[-1, 1089])
    
    return x_i_5
    
def ista_block(input_layers, input_data, layer_no, Weights0, Weights1, Weights2, Weights3, All_weights, J, shrinkage_thresh, N, xishu):
    
    alpha = tf.Variable(1.0, dtype=tf.float32, trainable=True)
    beta = tf.Variable(0.1, dtype=tf.float32, trainable=True)
    
    alpha = tf.abs(alpha)
    beta = tf.abs(beta)
    
    
    SSS = xishu*shrinkage_thresh

    
    xxx = input_layers[-1]
    
    # first step
    
    b_k = tf.add(xxx - tf.scalar_mul(alpha, tf.matmul(xxx, PhiTPhi)), tf.scalar_mul(alpha, PhiTb))  # X_k - lambda*A^TAX
            
    R_b_k = grad_R(b_k, Weights0, Weights1, Weights2, Weights3, SSS)
    
    u = b_k - beta * R_b_k
    
    
    
    
    R_x_k = grad_R(xxx, Weights0, Weights1, Weights2, Weights3, SSS)

    v = b_k - alpha*R_x_k
    
    f_b = 0.5 * tf.reduce_sum(tf.square(tf.matmul(u, Phi) - X_input), -1, keep_dims=True)
    f_x = 0.5 * tf.reduce_sum(tf.square(tf.matmul(v, Phi) - X_input), -1, keep_dims=True)
    
    
    RRR_u = R(u, Weights0, Weights1, Weights2, Weights3, SSS)

    cond_b = f_b + RRR_u
    
    RRR_v = R(v, Weights0, Weights1, Weights2, Weights3, SSS)

    cond_x = f_x + RRR_v
    
    sign_b_greater_than_x = tf.sign(tf.nn.relu(cond_b - cond_x))
    
    sign_out = tf.squeeze(sign_b_greater_than_x)
    
    sign_b_greater_than_x = tf.tile(sign_b_greater_than_x, [1, n_output])
    
    if N > 2:
        out = tf.cond(Epoch_i < 100, lambda: (tf.ones_like(sign_b_greater_than_x) - tf.zeros_like(sign_b_greater_than_x)) * u + tf.zeros_like(sign_b_greater_than_x) * v,
                  lambda: (tf.ones_like(sign_b_greater_than_x) - sign_b_greater_than_x) * u + sign_b_greater_than_x * v)
    else:
        out = tf.cond(Epoch_i < 200, lambda: (tf.ones_like(sign_b_greater_than_x) - tf.zeros_like(sign_b_greater_than_x)) * u + tf.zeros_like(sign_b_greater_than_x) * v,
                  lambda: (tf.ones_like(sign_b_greater_than_x) - sign_b_greater_than_x) * u + sign_b_greater_than_x * v)
    update_eta_coeff = tf.norm( tf.matmul(out, PhiTPhi) + PhiTb + grad_R(out, Weights0, Weights1, Weights2, Weights3, SSS), ord='euclidean', axis=-1)
    update_eta_coeff = tf.reduce_mean(update_eta_coeff)
    
    xishu = tf.cond(update_eta_coeff>= 15000.0 * SSS, lambda: xishu, lambda: 0.9*xishu)
    
    return [out, [tf.reduce_mean(cond_b), update_eta_coeff, SSS, xishu, shrinkage_thresh, alpha, beta, tf.reduce_mean(sign_out)], xishu]


def inference_ista(input_tensor, regu_num, n, N, X_output, reuse):
    layers = []
    layers.append(input_tensor)
    Sign_out = []
    All_weights = []
    for j in range(regu_num):
        if j == (regu_num - 1):
            [Weights0, bias0] = add_con2d_weight_bias([3, 3, 1, channel_number], [channel_number], 0)
            [Weights1, bias1] = add_con2d_weight_bias([3, 3, channel_number, channel_number], [channel_number], 1)
            [Weights2, bias2] = add_con2d_weight_bias([3, 3, channel_number, channel_number], [channel_number], 2)
            [Weights3, bias3] = add_con2d_weight_bias([3, 3, channel_number, channel_number], [channel_number], 3)
            shrinkage_thresh = tf.Variable(0.01, dtype=tf.float32, trainable=True)
            xishu = 1.0
            for i in range(n):
                with tf.variable_scope('conv_%d' %i, reuse=reuse):
                    [conv1, sign_out, xishu_out] = ista_block(layers, X_output, i, Weights0, Weights1, Weights2, Weights3, All_weights, j, shrinkage_thresh, n, xishu)
                    xishu = xishu_out
                    Sign_out.append(sign_out)
                    layers.append(conv1)
                
    return [layers, Sign_out]


def compute_cost(Prediction, X_output, PhaseNumber):
    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))

    return cost

best_epoch = []

for regu_num in range(1):
    
    regu_num = regu_num + 1
    
    for PhaseNumber in range(PhaseNumber_start, PhaseNumber_end, 2):
        
        if PhaseNumber > 12:
            batch_size = 32
        
        [Prediction, Sign_out] = inference_ista(X0, regu_num, PhaseNumber, PhaseNumber_end, X_output, reuse=tf.AUTO_REUSE)
    
        cost0 = tf.reduce_mean(tf.square(X0 - X_output))
        cost = compute_cost(Prediction, X_output, PhaseNumber)
    
        cost_all = cost
        
        
        optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam').minimize(cost_all)
        
        init = tf.global_variables_initializer()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        epoch_result = []
        sess = tf.Session(config=config)
        sess.run(init)
        if regu_num == 1 and PhaseNumber == PhaseNumber_start:
            print('start training')
        else:
            #qingchao added
            ISTA_logits = slim.get_variables_to_restore()
            init_Weights = 'Regu_%d_Phase_%d_ratio_0_%d_ISTA_Net_plus_Model-relu-cascade' % (regu_num, PhaseNumber-2, CS_ratio) + '/CS_Saved_Model_%d.cpkt' % (np.multiply(20, best_epoch[-1]))
            init_ISTA, init_feeddic = slim.assign_from_checkpoint(init_Weights, ISTA_logits, ignore_missing_vars = True)
            sess.run(init_ISTA, init_feeddic)
        
        print("...............................")
        print("Regulation number is %d, Phase Number is %d, CS ratio is %d%%" % (regu_num, PhaseNumber, CS_ratio))
        print("...............................\n")
        
        print("Strart Training..")
        
        
        model_dir = 'Regu_%d_Phase_%d_ratio_0_%d_ISTA_Net_plus_Model-relu-cascade' % (regu_num, PhaseNumber, CS_ratio)
        
        output_file_name = "Log_output_%s.txt" % (model_dir)
        
        if PhaseNumber > 3:
            EpochNum = 200
        else:
            EpochNum = 500
        
        for epoch_i in range(0, EpochNum+1):
            randidx_all = np.random.permutation(nrtrain)
            for batch_i in range(nrtrain // batch_size):
                randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
        
                batch_ys = Training_labels[randidx, :]
                batch_xs = np.dot(batch_ys, Phi_input)
                nnn = batch_ys.shape[0]
        
                feed_dict = {X_input: batch_xs, X_output: batch_ys, Epoch_i: epoch_i}
                sess.run(optm_all, feed_dict=feed_dict)
        
            output_data = "[%02d/%02d] cost: %.4f\n" % (epoch_i, EpochNum, sess.run(cost, feed_dict=feed_dict))
            print(output_data)
        
            output_file = open(output_file_name, 'a')
            output_file.write(output_data)
            output_file.close()
        
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if epoch_i <= 15:
                saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
                PSNR_All, ERROR_All = rec(epoch_i, PhaseNumber, regu_num, Prediction)
            else:
                if epoch_i % 20 == 0:
                    saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
                    PSNR_All, ERROR_All = rec(epoch_i, PhaseNumber, regu_num, Prediction)
                    epoch_result.append(PSNR_All)
        best_epoch.append(np.where(epoch_result==np.max(epoch_result)))
        
        print("Training Finished")
        sess.close()
            

    
    