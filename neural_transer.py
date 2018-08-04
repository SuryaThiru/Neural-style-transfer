import keras.backend as K
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from PIL import Image
import numpy as np


# Utilities

def imsave(img, path, target_size=(512, 512), postprocess=True):
    if postprocess:
        img = postprocess_array(img)
    img = Image.fromarray(img)
    img = img.resize(target_size)
    img.save(path)
    return img


def imread_tensor(path, target_size=(512, 512)):
    '''
    reads an image and returns a preprocessed tensor
    '''
    img = load_img(path=path, target_size=target_size)
    img = img_to_array(img)
    img = K.variable(preprocess_input(np.expand_dims(img, axis=0)),
                     dtype='float32')
    return img


def postprocess_array(x, target_size=(512, 512, 3)):
    if x.shape != target_size:
        x = x.reshape(target_size)
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68

    # BGR to RGB
    x = x[..., ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Important operations

def generate_canvas(mode='random', ref_image=None):
    '''
    Generate a canvas and return a placeholder
    Params:
    modes: random, from_ref
    ref_image: pass an img array or path if mode is 'from_ref'
    '''
    size = (512, 512, 3)

    if mode == 'random':
        img = np.random.randint(256, size=size)
    elif mode == 'from_ref':
        if type(ref_image) == str:
            img = load_img(path=ref_image, target_size=size)
            img = img_to_array(img)
        else:
            img = ref_image.copy()

    img = preprocess_input(np.expand_dims(img, axis=0))

    return img


def get_feature_maps(model, layers, tf_session):
    '''
    Get feature maps for given layers in the required format
    '''
    features = []

    for layer in layers:
        feat = model.get_layer(layer).output
        shape = K.shape(feat).eval(session=tf_session)
        M = shape[1] * shape[2]
        N = shape[-1]
        feat = K.transpose(K.reshape(feat, (M, N)))
        features.append(feat)

    return features


def content_loss(F, P):
    assert F.shape == P.shape
    loss = 0.5 * K.sum(K.square(F - P))
    return loss


def gram_matrix(matrix):
    return K.dot(matrix, K.transpose(matrix))


def style_loss(G, A):
    ''' Contribution of each layer to the total style loss
    '''
    assert G.shape == A.shape

    M, N = K.int_shape(G)[1], K.int_shape(G)[0]
    G, A = gram_matrix(G), gram_matrix(A)
    loss = 0.25 * K.sum(K.square(G - A)) / ((N ** 2) * (M ** 2))
    return loss


def total_style_loss(weights, Gs, As):
    ''' Get weighted total style loss
    '''
    loss = K.variable(0)

    for w, G, A in zip(weights, Gs, As):
        loss = loss + w * style_loss(G, A)

    return loss


def total_loss(P, As, canvas_model, clayers, slayers, style_weights, tf_session, alpha=1.0, beta=10000.0):
    '''
    Get total loss
    Params:
    x: generated image
    p: content image features
    a: style image features
    '''
    F = get_feature_maps(canvas_model, clayers, tf_session)[0]
    Gs = get_feature_maps(canvas_model, slayers, tf_session)

    closs = content_loss(F, P)
    sloss = total_style_loss(style_weights, Gs, As)

    loss = alpha * closs + beta * sloss
    return loss


step = 1


def style_transfer(cnt_img_path, style_img_path, output_path='output/', epochs=50, save_per_epoch=20):
    target_size = (512, 512, 3)

    cnt_img = imread_tensor(cnt_img_path)
    style_img = imread_tensor(style_img_path)
    canvas_placeholder = K.placeholder(shape=(1,) + target_size)

    cnt_model = VGG16(include_top=False, weights='imagenet',
                      input_tensor=cnt_img)
    style_model = VGG16(include_top=False, weights='imagenet',
                        input_tensor=style_img)
    canvas_model = VGG16(include_top=False, weights='imagenet',
                         input_tensor=canvas_placeholder)

    tf_session = K.get_session()

    cnt_layers = ['block4_conv2']
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
    ]
    # weights for each style layer
    Ws = [1.0 / float(len(style_layers))] * len(style_layers)

    P = get_feature_maps(cnt_model, cnt_layers, tf_session)[0]
    As = get_feature_maps(style_model, style_layers, tf_session)

    # generate canvas from content
    X = generate_canvas('from_ref', cnt_img_path).flatten()

    def calculate_loss(gimg):
        gimg = gimg.reshape((1,) + target_size)

        loss = total_loss(P, As, canvas_model, cnt_layers,
                          style_layers, Ws, tf_session=tf_session)
        loss_func = K.function([canvas_model.input], [loss])
        return loss_func([gimg])[0].astype('float64')

    def calculate_grad(gimg):
        gimg = gimg.reshape((1,) + target_size)

        loss = total_loss(P, As, canvas_model, cnt_layers,
                          style_layers, Ws, tf_session=tf_session)
        gradients = K.gradients(loss, [canvas_model.input])
        grad_func = K.function([canvas_model.input], gradients)
        return grad_func([gimg])[0].flatten().astype('float64')

    def callback(gimg):
        global step

        print(f'\rStep: {step}/{epochs}', end='')
        step += 1

        if (step % save_per_epoch) == 0 or (step == epochs):
            gimg = gimg.copy()
            path = output_path + f'out_{step}.jpg'
            imsave(gimg, path)

    print('Optimizing...\n')

    X_optim, _, info = fmin_l_bfgs_b(
        calculate_loss, X, fprime=calculate_grad,
        maxiter=epochs, callback=callback)

    print('Saving final generated image...')
    path = output_path + 'optimal.jpg'
    imsave(X_optim, path)
