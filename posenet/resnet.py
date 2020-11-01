import os
import tensorflow as tf
import shutil
import urllib.request
import json
import zlib
import posixpath
from abc import ABC, abstractmethod
import cv2
import numpy as np
import tfjs_graph_converter as tfjs
import posenet
BASE_DIR = os.path.dirname(__file__)
TFJS_MODEL_DIR = './_tfjs_models'
TF_MODEL_DIR = './_tf_models'

RESNET50_BASE_URL = 'https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/'

POSENET_ARCHITECTURE = 'posenet'

RESNET50_MODEL = 'resnet50'
class BaseModel(ABC):

    # keys for the output_tensor_names map
    HEATMAP_KEY = "heatmap"
    OFFSETS_KEY = "offsets"
    DISPLACEMENT_FWD_KEY = "displacement_fwd"
    DISPLACEMENT_BWD_KEY = "displacement_bwd"

    def __init__(self, model_function, output_tensor_names, output_stride):
        self.output_stride = output_stride
        self.output_tensor_names = output_tensor_names
        self.model_function = model_function

    def valid_resolution(self, width, height):
        # calculate closest smaller width and height that is divisible by the stride after subtracting 1 (for the bias?)
        target_width = (int(width) // self.output_stride) * self.output_stride + 1
        target_height = (int(height) // self.output_stride) * self.output_stride + 1
        return target_width, target_height

    @abstractmethod
    def preprocess_input(self, image):
        pass

    def predict(self, image):
        input_image, image_scale = self.preprocess_input(image)

        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

        result = self.model_function(input_image)

        heatmap_result = result[self.output_tensor_names[self.HEATMAP_KEY]]
        offsets_result = result[self.output_tensor_names[self.OFFSETS_KEY]]
        displacement_fwd_result = result[self.output_tensor_names[self.DISPLACEMENT_FWD_KEY]]
        displacement_bwd_result = result[self.output_tensor_names[self.DISPLACEMENT_BWD_KEY]]

        return tf.sigmoid(heatmap_result), offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale

class ResNet(BaseModel):

    def __init__(self, model_function, output_tensor_names, output_stride):
        super().__init__(model_function, output_tensor_names, output_stride)
        self.image_net_mean = [-123.15, -115.90, -103.06]

    def preprocess_input(self, image):
        target_width, target_height = self.valid_resolution(image.shape[1], image.shape[0])
        # the scale that can get us back to the original width and height:
        scale = np.array([image.shape[0] / target_height, image.shape[1] / target_width])
        input_img = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)  # to RGB colors
        # todo: test a variant that adds black bars to the image to match it to a valid resolution

        # See: https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/resnet.ts
        input_img = input_img + self.image_net_mean
        input_img = input_img.reshape(1, target_height, target_width, 3)  # HWC to NHWC
        return input_img, scale
class PoseNet:

    def __init__(self, model: BaseModel, min_score=0.25):
        self.model = model
        self.min_score = min_score

    def estimate_multiple_poses(self, image, max_pose_detections=10):
        heatmap_result, offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale = \
            self.model.predict(image)

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmap_result.numpy().squeeze(axis=0),
            offsets_result.numpy().squeeze(axis=0),
            displacement_fwd_result.numpy().squeeze(axis=0),
            displacement_bwd_result.numpy().squeeze(axis=0),
            output_stride=self.model.output_stride,
            max_pose_detections=max_pose_detections,
            min_pose_score=self.min_score)

        keypoint_coords *= image_scale

        return pose_scores, keypoint_scores, keypoint_coords

    def estimate_single_pose(self, image):
        return self.estimate_multiple_poses(image, max_pose_detections=1)

    def draw_poses(self, image, pose_scores, keypoint_scores, keypoint_coords):
        draw_image = posenet.draw_skel_and_kp(
            image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=self.min_score, min_part_score=self.min_score)

        return draw_image

    def print_scores(self, image_name, pose_scores, keypoint_scores, keypoint_coords):
        print()
        print("Results for image: %s" % image_name)
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
def fix_model_file(model_cfg):
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])

    if not model_cfg['filename'] == 'model.json':
        # The expected filename for the model json file is 'model.json'.
        # See tfjs_common.ARTIFACT_MODEL_JSON_FILE_NAME in the tensorflowjs codebase.
        normalized_model_json_file = os.path.join(model_cfg['tfjs_dir'], 'model.json')
        shutil.copyfile(model_file_path, normalized_model_json_file)

    with open(model_file_path, 'r') as f:
        json_model_def = json.load(f)

    return json_model_def


def download_single_file(base_url, filename, save_dir):
    output_path = os.path.join(save_dir, filename)
    url = posixpath.join(base_url, filename)
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    if response.info().get('Content-Encoding') == 'gzip':
        data = zlib.decompress(response.read(), zlib.MAX_WBITS | 32)
    else:
        # this path not tested since gzip encoding default on google server
        # may need additional encoding/text handling if hit in the future
        data = response.read()
    with open(output_path, 'wb') as f:
        f.write(data)


def download_tfjs_model(model_cfg):
    """
    Download a tfjs model with saved weights.
    :param model_cfg: The model configuration
    """
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])
    if os.path.exists(model_file_path):
        print('Model file already exists: %s...' % model_file_path)
        return
    if not os.path.exists(model_cfg['tfjs_dir']):
        os.makedirs(model_cfg['tfjs_dir'])

    download_single_file(model_cfg['base_url'], model_cfg['filename'], model_cfg['tfjs_dir'])

    json_model_def = fix_model_file(model_cfg)

    shard_paths = json_model_def['weightsManifest'][0]['paths']
    for shard in shard_paths:
        download_single_file(model_cfg['base_url'], shard, model_cfg['tfjs_dir'])

def bodypix_resnet50_config(stride, quant_bytes=4):

    graph_json = 'model-stride' + str(stride) + '.json'

    # quantBytes = 4 corresponding to the non - quantized full - precision checkpoints.
    if quant_bytes == 4:
        base_url = RESNET50_BASE_URL + 'float'
        model_dir = RESNET50_MODEL + '_float'
    else:
        base_url = RESNET50_BASE_URL + 'quant' + str(quant_bytes) + '/'
        model_dir = RESNET50_MODEL + '_quant' + str(quant_bytes)

    stride_dir = 'stride' + str(stride)

    return {
        'base_url': base_url,
        'filename': graph_json,
        'output_stride': stride,
        'data_format': 'NHWC',
        'input_tensors': {
            'image': 'sub_2:0'
        },
        'output_tensors': {
            'heatmap': 'float_heatmaps:0',
            'offsets': 'float_short_offsets:0',
            'displacement_fwd': 'resnet_v1_50/displacement_fwd_2/BiasAdd:0',
            'displacement_bwd': 'resnet_v1_50/displacement_bwd_2/BiasAdd:0'
        },
        'tfjs_dir': os.path.join(TFJS_MODEL_DIR, POSENET_ARCHITECTURE, model_dir, stride_dir),
        'tf_dir': os.path.join(TF_MODEL_DIR, POSENET_ARCHITECTURE, model_dir, stride_dir)
    }
def __tensor_info_def(sess, tensor_names):
    signatures = {}
    for tensor_name in tensor_names:
        tensor = sess.graph.get_tensor_by_name(tensor_name)
        tensor_info = tf.compat.v1.saved_model.build_tensor_info(tensor)
        signatures[tensor_name] = tensor_info
    return signatures


def convert(model_cfg):
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])
    if not os.path.exists(model_file_path):
        print('Cannot find tfjs model path %s, downloading tfjs model...' % model_file_path)
        download_tfjs_model(model_cfg)

    # 'graph_model_to_saved_model' doesn't store the signature for the model!
    #   tfjs.api.graph_model_to_saved_model(model_cfg['tfjs_dir'], model_cfg['tf_dir'], ['serve'])
    # So we do it manually below.
    # This link was a great help to do this:
    # https://www.programcreek.com/python/example/104885/tensorflow.python.saved_model.signature_def_utils.build_signature_def

    graph = tfjs.api.load_graph_model(model_cfg['tfjs_dir'])
    builder = tf.compat.v1.saved_model.Builder(model_cfg['tf_dir'])

    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor_names = tfjs.util.get_input_tensors(graph)
        output_tensor_names = tfjs.util.get_output_tensors(graph)

        signature_inputs = __tensor_info_def(sess, input_tensor_names)
        signature_outputs = __tensor_info_def(sess, output_tensor_names)

        method_name = tf.compat.v1.saved_model.signature_constants.PREDICT_METHOD_NAME
        signature_def = tf.compat.v1.saved_model.build_signature_def(inputs=signature_inputs,
                                                                     outputs=signature_outputs,
                                                                     method_name=method_name)
        signature_map = {tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def}
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=['serve'],
                                             signature_def_map=signature_map)
    return builder.save()

def load_model(model, stride, quant_bytes=4, multiplier=1.0):

    model = RESNET50_MODEL
    model_cfg = bodypix_resnet50_config(stride, quant_bytes)
    print('Loading ResNet50 model')

    model_path = model_cfg['tf_dir']
    if not os.path.exists(model_path):
        print('Cannot find tf model path %s, converting from tfjs...' % model_path)
        convert(model_cfg)
        assert os.path.exists(model_path)

    loaded_model = tf.saved_model.load(model_path)

    signature_key = tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    print('We use the signature key %s It should be in the keys list:' % signature_key)
    for sig in loaded_model.signatures.keys():
        print('signature key: %s' % sig)

    model_function = loaded_model.signatures[signature_key]
    print('model outputs: %s' % model_function.structured_outputs)

    output_tensor_names = model_cfg['output_tensors']
    output_stride = model_cfg['output_stride']

    net = ResNet(model_function, output_tensor_names, output_stride)


    return PoseNet(net)