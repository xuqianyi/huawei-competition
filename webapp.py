import os
import uuid
import urllib
from typing import Tuple

import cv2
import flask
import numpy as np
import spacy
import tensorflow as tf
from keras.backend import set_session
from keras.layers import Input
from yacs.config import CfgNode as CN

from loader.loader import qlist_to_vec
from model.vlt_model import yolo_body

OUT_PATH = 'log/web_out'
CONFIG_PATH = 'config/refcoco/example.yaml'

with open('config/base.yaml', 'r') as f:
    _C = CN.load_cfg(f)

config = _C.clone()
config.merge_from_file(CONFIG_PATH)
config.freeze()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(config.seed)
tf.set_random_seed(config.seed)


class WebHost():
    def __init__(self, config: CN, sess: tf.Session):
        self.config = config
        self.GPUS = 1
        self.input_shape = (self.config.input_size, self.config.input_size, 3)
        self.word_len = self.config.word_len
        self.embed_dim = self.config.embed_dim
        self.session = sess

        set_session(sess)

        print('Creating model...')
        image_input = Input(shape=(self.input_shape))
        q_input = Input(shape=[self.word_len, self.embed_dim], name='q_input')

        self.model = yolo_body(image_input, q_input, self.config)
        print('Loading model...')
        self.model.load_weights(config.evaluate_model, by_name=False, skip_mismatch=False)
        print('Load weights {}.'.format(config.evaluate_model))
        self.graph = tf.get_default_graph()

        self.embed = spacy.load(config.word_embed)

    def preprocess(self, image: np.ndarray, lang: str) -> Tuple[np.ndarray, np.ndarray]:
        h, w, _ = self.input_shape
        ih, iw, _ = image.shape

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        image_data = np.full((w, h, 3), (0.5, 0.5, 0.5))
        image_data[dy:dy+nh, dx:dx+nw, :] = image / 255.

        word_vec = qlist_to_vec(self.word_len, lang, self.embed)

        return image_data, word_vec

    def predict(self, image: np.ndarray, lang: str) -> np.ndarray:
        img, word_vec = self.preprocess(image, lang)
        with self.graph.as_default():
            set_session(self.session)
            output = self.model.predict([[img], [word_vec]])
        output = self.sigmoid_(output)
        return output

    def sigmoid_(self, x: np.ndarray) -> np.ndarray:
        return (1. + 1e-9) / (1. + np.exp(-x) + 1e-9)


sess = tf.Session()
KerasHost = WebHost(config, sess)
app = flask.Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}

    if flask.request.files.get('image'):
        image = flask.request.files['image'].read()
        file_bytes = np.asarray(bytearray(image), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    lang = flask.request.form['lang']

    output = KerasHost.predict(img, lang)

    request_id = str(uuid.uuid4())
    filename = '{}.png'.format(request_id)
    success = cv2.imwrite(os.path.join(OUT_PATH, filename), output.squeeze() * 255)

    data['success'] = success
    if success:
        img_url = urllib.parse.urljoin(flask.request.host_url, os.path.join('outputs', filename))
        data['img_url'] = img_url

    return flask.jsonify(data)


@app.route('/outputs/<path:path>', methods=['GET'])
def send_output_img(path):
    if not path.endswith('.png'):
        return 'Illigial Filename', 400
    if not os.path.exists(os.path.join(OUT_PATH, path)):
        return 'File Not Found', 400
    return flask.send_from_directory(OUT_PATH, path)


if __name__ == '__main__':
    app.run()
