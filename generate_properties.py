'''
vggnet16 vnncomp 2023 benchmark

Stanley Bak
'''

import sys
import os
import time
import random

import numpy as np
import onnx
import onnxruntime as ort

# first run pip3 install mxnet if needed
import subprocess
try:
    import mxnet as mx
except ImportError:
    print('pip installing mxnet in current environment in 5 seconds (ctrl-c to cancel)')
    time.sleep(5)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mxnet'])

import mxnet as mx
from mxnet.gluon.data.vision import transforms

def predict_with_onnxruntime(sess, *inputs):
    'run an onnx model'
    
    names = [i.name for i in sess.get_inputs()]

    inp = dict(zip(names, inputs))

    res = sess.run(None, inp)

    #names = [o.name for o in sess.get_outputs()]

    return res[0]

def get_io_nodes(onnx_model, sess):
    'returns 3 -tuple: input node, output nodes, input dtype'

    #sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = [i.name for i in sess.get_inputs()]
    assert len(inputs) == 1, f"expected single onnx network input, got: {inputs}"
    input_name = inputs[0]

    outputs = [o.name for o in sess.get_outputs()]
    assert len(outputs) == 1, f"expected single onnx network output, got: {outputs}"
    output_name = outputs[0]

    g = onnx_model.graph
    inp = [n for n in g.input if n.name == input_name][0]
    out = [n for n in g.output if n.name == output_name][0]

    input_type = g.input[0].type.tensor_type.elem_type

    assert input_type in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]

    dtype = np.float32 if input_type == onnx.TensorProto.FLOAT else np.float64

    return inp, out, dtype

def normalize(img):
    """apply vggnet normalization"""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for channel in range(3):
        img[:, :, channel] = (img[:, :, channel] - mean[channel]) / std[channel]

    return img

def make_input(image_filename, inp_shape):
    """make input tensor"""

    img = mx.image.imread(image_filename)

    # original:
    #transform_fn = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    #transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #])

    # I (Stan) took this from: https://huggingface.co/spaces/onnx/VGG/blob/main/app.py
    # on the 1000 sample images the accuracty I get is slightly different than reported in the VGGNET paper
    # got: top 1: 57% error, top 5: 4% error
    # expected: top 1: 25% error, top 5: 8% error
    # I blame the difference on the images not being representative of all ImageNet images, although I didn't test this

    transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = transform_fn(img)
    img = img.expand_dims(axis=0)

    img = img.asnumpy()

    assert img.shape == inp_shape, f"image.shape: {img.shape}, inp_shape: {inp_shape}"

    return img

def make_spec(spec_index, onnx_filename, image_index, image_filename, spec_path):
    '''execute the model and its conversion as a sanity check

    returns string to print to output file
    '''

    start = time.perf_counter()
    onnx_model = onnx.load(onnx_filename)

    mid = time.perf_counter()
    load_time = mid - start
    print(f"load time: {load_time}")
    
    sess = ort.InferenceSession(onnx_filename)
    
    session_time = time.perf_counter() - mid
    print(f"session time: {session_time}")
    
    #onnx.checker.check_model(onnx_model, full_check=True)
    #onnx_model = remove_unused_initializers(onnx_model)

    inp, out, inp_dtype = get_io_nodes(onnx_model, sess)
    
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    print(f"inp_shape: {inp_shape}")
    print(f"out_shape: {out_shape}")

    num_inputs = 1
    num_outputs = 1

    for n in inp_shape:
        num_inputs *= n

    for n in out_shape:
        num_outputs *= n

    print(f"Testing onnx model with {num_inputs} inputs and {num_outputs} outputs")

    top1 = []
    top5 = []
    total = 0

    input_tensor = make_input(image_filename, inp_shape)

    output = predict_with_onnxruntime(sess, input_tensor)

    out_flat = output.flatten('C') # double-check order
    in_flat = input_tensor.flatten('C')

    #index = np.argmax(out_flat)
    top5_inds = list(reversed(np.argpartition(out_flat, -5)[-5:]))

    #output_dict = get_output_dict()
    if top5_inds[0] != image_index:
        print(f'top1 was incorrect, got {top5_inds[0]} expected {image_index}')
            
        return ''

    # result was correct, produce the spec file
    pixel_index = (spec_index // 3) % 6
    print(f"spec_index: {spec_index}, pixel_index: {pixel_index}")
    num_pixels = [1, 5, 10, 20, 100, 150528][pixel_index]

    if num_pixels < 5000:
        perturb_eps = [1e-5, 1e-4, 1e-3][spec_index % 3]
    else:
        perturb_eps = [1e-7, 1e-6, 1e-5][spec_index % 3]

    perturb_pixels = set(random.sample(range(150528), num_pixels))

    print(f"perturbing {num_pixels} pixels by {perturb_eps}")

    with open(spec_path, 'w', encoding='utf-8') as f:
        f.write(f'; VGGNET Spec for image {image_index}: {image_filename}\n\n')

        for i in range(num_inputs):
            f.write(f'(declare-const X_{i} Real)\n')

        f.write('\n')

        for i in range(1000):
            f.write(f'(declare-const Y_{i} Real)\n')

        f.write('\n; Input constraints:\n')

        assert len(in_flat) == num_inputs

        #for channel in range(3):
        #    single_channel = input_tensor[:,channel,:,:].flatten()
            # print min and max of in_flat
        #    print(f"channel {channel}, min: {np.min(single_channel)}, max: {np.max(single_channel)}")
            
        #exit(1)

        for index, x in enumerate(in_flat):

            if index in perturb_pixels:
                eps = perturb_eps
            else:
                eps = 0
            
            # maybe we should trim x +/- eps to limits
            f.write(f'(assert (<= X_{index} {x + eps}))\n')
            f.write(f'(assert (>= X_{index} {x - eps}))\n\n')

        # targetted misclasification
        f.write('\n; Output constraints (encoding the conditions for a property counter-example):\n')

        top1 = top5_inds[0]
        any_cat = False # spec type: target category or any category

        if any_cat:
            f.write('(assert (or\n')

            for i in range(1000):
                if i == top1:
                    continue

                f.write(f' (and (>= Y_{i} Y_{top1}))\n')

            f.write('))\n')
        else:
            # single-cat
            top2 = top5_inds[1]
            f.write(f'(assert (>= Y_{top2} Y_{top1}))\n')

    print(f'wrote: {spec_path}')

    return True


def get_image_paths(image_dir):
    """get 1000 paths to images"""

    paths = []

    for path in os.listdir(image_dir):

        if 'JPEG' not in path:
            continue
        
        fullpath = os.path.join(image_dir, path)
        
        if os.path.isfile(fullpath):
            paths.append(fullpath)

    paths.sort()

    assert len(paths) == 1000

    return paths

def main():
    """main entry point"""

    assert len(sys.argv) == 2, "expected 1 arg: <seed>"
    random.seed(int(sys.argv[1]))

    # prepare vnnlib and onnx directories
    for dirname in ['vnnlib', 'onnx']:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        elif dirname == 'vnnlib':
            for filename in os.listdir(dirname):
                os.remove(os.path.join(dirname, filename))

    # download vggnet 16 if needed
    if not os.path.exists('onnx/vgg16-7.onnx'):
        os.system("wget https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/validated/vision/classification/vgg/model/vgg16-7.onnx -O onnx/vgg16-7.onnx")

    onnx_filename = 'onnx/vgg16-7.onnx'
    image_dir = "imagenet-sample"

    image_paths = get_image_paths(image_dir)
    
    num_images = 0
    total_images = 18 # 20 minute timeout each for 6 hours

    with open('instances.csv', 'w', encoding='utf-8') as f:

        while num_images < total_images:
            image_index = random.randint(0, 1000)
            image_filename = image_paths[image_index]

            print(f"trying image index {image_index}: {image_filename}")

            left_index = 1 + image_filename.index('_')
            right_index = 1 + image_filename.index('.') - 1
            name = image_filename[left_index:right_index]
            print(name)

            spec_path = f'vnnlib/spec{num_images}_{name}.vnnlib'

            spec_index = num_images
            made_spec = make_spec(spec_index, onnx_filename, image_index, image_filename, spec_path)

            if made_spec:
                f.write(f'{onnx_filename},{spec_path},1200\n')
                    
                num_images += 1
                print(f"wrote {num_images} / {total_images}\n")

if __name__ == '__main__':
    main()
