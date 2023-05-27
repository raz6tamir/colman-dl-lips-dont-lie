import multiprocessing as mp
import pathlib
import random
from pathlib import Path

import cv2
import dlib
import keras
import numpy as np
import tensorflow as tf
from PIL import Image

# model = keras.models.load_model('model')

root_path = Path.cwd()


def load_file(file_path):
    data = np.load(file_path)
    return data['data']


def expand_image_channels(image):
    image_arr = np.array(image)
    image_arr = np.expand_dims(image_arr, 1)
    return image_arr


class DataGenerator:

    def __init__(self, path, set_type):
        """ Returns a set of frames with their associated label.

          Args:
            path: file paths (directory path).
            set_type: type of set to create - train, val or test.
        """
        self.path = path
        self.set_type = set_type
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        data_paths = list(self.path.glob(f'**/{self.set_type}/*.npz'))
        classes = [p.parent.parent.name for p in data_paths]
        return data_paths, classes

    def __call__(self):
        data_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(data_paths, classes))

        if self.set_type == 'train':
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = load_file(path)
            video_frames = expand_image_channels(video_frames)
            label = self.class_ids_for_name[name]  # Encode labels
            yield video_frames, label - 1


def get_eyes_nose_dlib(shape):
    nose = (shape.part(34).x, shape.part(34).y)
    left_eye_x = int(shape.part(37).x + shape.part(40).x) // 2
    left_eye_y = int(shape.part(37).y + shape.part(40).y) // 2
    right_eyes_x = int(shape.part(43).x + shape.part(46).x) // 2
    right_eyes_y = int(shape.part(43).y + shape.part(46).y) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)


def get_mouth_dlib(shape):
    x = shape.part(48).x
    y = shape.part(48).y
    w = shape.part(54).x - shape.part(48).x
    h = shape.part(51).y - shape.part(57).y
    return (x, y, w, h)


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a


def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (
            extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (
            extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (
            extra_point[0] - point3[0])
    return ((c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0))


def align_face_img(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('landmarks_predictor/shape_predictor_68_face_landmarks.dat')

    img = np.asarray(img)
    img_x = img.shape[1] // 2
    img_y = img.shape[0] // 2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # retrive the coordinates of a face rectangle
    # and the shape of the facial landmarks (eyes and nose coordinates)
    rects = detector(gray, 1)
    if len(rects) > 0:
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
            shape = predictor(gray, rect)

        # obtain the central coordinates of nose and eyes
        nose, left_eye, right_eye = get_eyes_nose_dlib(shape)

        # find the center of the line between two eyes (endpoint of the median)
        center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

        # find the center of the face rectangle top side
        center_pred = (int((x + w) / 2), int((y + y) / 2))

        # length_line1(the median), length_line2 — lines between which we need to find the angle
        length_line1 = distance(center_of_forehead, nose)
        length_line2 = distance(center_pred, nose)
        length_line3 = distance(center_pred, center_of_forehead)

        # retrieve the angle in radians
        cos_a = cosine_formula(length_line1, length_line2, length_line3)
        angle = np.arccos(cos_a)

        # find rotation angle
        rotated_point = rotate_point(nose, center_of_forehead, angle)
        rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
        if is_between(nose, center_of_forehead, center_pred, rotated_point):
            dg_angle = np.degrees(-angle)
        else:
            dg_angle = np.degrees(angle)

        img = Image.fromarray(img)
        img = img.rotate(dg_angle)

        mouth_x, mouth_y, mouth_w, mouth_h = get_mouth_dlib(shape)
        rotated_x, rotated_y = rotate_point((img_x, img_y), (mouth_x, mouth_y), angle)

        rotated_x = int(rotated_x)
        rotated_y = int(rotated_y)

        y_pad = (76 - mouth_h) // 2
        x_pad = (76 - mouth_w) // 2

        cropped_img = np.array(img)[rotated_y - y_pad - (mouth_h // 2):rotated_y + (mouth_h // 2) + y_pad,
                      rotated_x - x_pad:rotated_x + mouth_w + x_pad]
        cropped_img = cv2.resize(cropped_img, (96, 96))

        return (img, Image.fromarray(cropped_img))
    else:
        raise Exception()


def extract_frames(video_path, video_name):
    frames = []
    cap = cv2.VideoCapture(video_path)
    print(f'\033[90m\033[0m{video_name}', end='\t')

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is not True:
            break
        frames.append(frame)
    cap.release()

    frames2return = []

    try:
        # with mp.Pool(29) as pool:
        results = [align_face_img(f,) for f in frames]

        for r in results:
            rotadet_img, cropped_img = r
            frames2return.append(cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2GRAY))

        print("DONE")
        return frames2return

    except Exception as e:
        raise e


def directory_process(rg_list_, log_path='log.txt'):
    with open(log_path, 'a+') as log_file:
        for file_path in rg_list_:
            file_name = '/'.join(str(file_path).split('/')[-1:])
            try:
                frames = extract_frames(str(file_path), file_name)
                npz_path = f"{root_path}/predict_ds/predict/test/input.npz"
                pathlib.Path('/'.join(npz_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
                np.savez(npz_path, data=frames)
            except Exception as e:
                print(f'\033[91m\033[1mError - {file_name}')
                log_file.write(f'{file_name}\n')
                raise


def run_model():
    # rg = pathlib.Path(root_path).rglob('video.mp4')
    # rg_list = list(rg)
    # print(len(rg_list))
    #
    # directory_process(rg_list)
    #
    # a = np.load(f"{root_path}/predict_ds/predict/test/input.npz")
    #
    # print(a['data'].shape)
    #
    # middle = a['data'].shape[0] // 2
    # trimmed_video = a['data'][middle - 20:middle + 9]
    # print(trimmed_video.shape)
    #
    # np.savez(f"{root_path}/predict_ds/predict/test/input.npz", data=trimmed_video)
    #
    # output_signature = (tf.TensorSpec(shape=(29, 1, 96, 96), dtype=tf.float32),
    #                     tf.TensorSpec(shape=(), dtype=tf.int16))
    #
    # predict_ds = tf.data.Dataset.from_generator(DataGenerator(pathlib.Path(f"{root_path}/predict_ds/"), set_type='test'),
    #                                             output_signature=output_signature)
    # predict_ds = predict_ds.batch(1)
    #
    # for frames, labels in predict_ds.take(1):
    #     print(f"Shape: {frames.shape}")
    #     print(f"Label: {labels.shape}")
    #
    # label_dict = {'ACTION': 1, 'CLOSE': 2, 'HOSPITAL': 3, 'LITTLE': 4, 'NUMBER': 5, 'PARTY': 6, 'RESULT': 7, 'SEVEN': 8,
    #               'TOMORROW': 9, 'WALES': 10}
    #
    # print(label_dict)
    #
    # from collections import defaultdict
    # labels_from_key = defaultdict(list)
    # for k, v in label_dict.items():
    #     labels_from_key[v].append(k)
    #
    # predicted = model.predict(predict_ds)
    #
    # print(predicted)
    #
    # predicted = tf.argmax(predicted, axis=1)
    # print(predicted[0].numpy())
    #
    # return labels_from_key[predicted[0].numpy()]

    return 'ACTION'
