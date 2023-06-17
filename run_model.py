import shutil

import multiprocessing as mp
import os
import pathlib
from pathlib import Path

import cv2
import dlib
import numpy as np
import requests
from PIL import Image

root_path = Path.cwd()


def image_resize(image, width=None, height=None):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim)

    # return the resized image
    return resized


# DLib Functions
# --------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f'{root_path}/landmarks_predictor/shape_predictor_68_face_landmarks.dat')


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_mouth_jaw_dlib_all(shape):
    mouth = [(shape.part(i).x, shape.part(i).y) for i in range(48, 68)]
    jaw = [(shape.part(i).x, shape.part(i).y) for i in range(17)]
    return mouth  # + jaw


# cast facial landmarks to a normal representation

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


# In order to find the angle between two sides of a triangle,
# knowing three of them, we can use a formula from a cosine rule

def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a


# in order to understand what the final angle we will use to rotate our image is,
# we need to rotate the end point of a median and check if it belongs to the space of the second triangle.
# In order to cope with it, we will use this functions

# rotates point by an angle around the origin
def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


# when given three tops of the triangle and one extra_point
# checks if the extra point lies in the space of the triangle
def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    return ((c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0))


# Extract words
# -------------
def get_mouth_points_from_frame(frame):
    img = np.asarray(frame)
    img_resized = img = image_resize(img, width=630)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # retrive the coordinates of a face rectangle
    # and the shape of the facial landmarks
    rects = detector(gray, 1)
    if len(rects) > 0:
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()
            shape = predictor(gray, rect)

        all_mouth_points = get_mouth_jaw_dlib_all(shape)

        return all_mouth_points
    else:
        return []


def get_words_from_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is not True:
            break
        frames.append(frame)
    cap.release()

    mouth_points_arr = []

    # with mp.Pool(29) as pool:
    #     results = [pool.apply_async(get_mouth_points_from_frame, (f,)) for f in frames]
    #
    #     for r in results:
    #         frame_mouth_points = r.get()
    #         if len(frame_mouth_points):
    #             mouth_points_arr.append(frame_mouth_points)

    results = [get_mouth_points_from_frame(f) for f in frames]

    for r in results:
        frame_mouth_points = r
        if len(frame_mouth_points):
            mouth_points_arr.append(frame_mouth_points)

    diff_arr = []

    for i in range(len(mouth_points_arr) - 1):
        sum_dist = 0
        for j in range(len(mouth_points_arr[i])):
            sum_dist += (distance(mouth_points_arr[i][j], mouth_points_arr[i + 1][j]) ** 2)
        diff_arr.append(sum_dist / len(mouth_points_arr[i]))

    threshold = 2
    stillness_frames = 10

    over_threshold = np.where(np.asarray(diff_arr) > threshold)[0]
    start_frame = max(0, over_threshold[0] - 2)
    i = start_frame + 1
    still_count = 0
    words = []

    while i < len(diff_arr):
        if diff_arr[i] < threshold:
            still_count += 1
        else:
            still_count = 0

        if still_count == stillness_frames:
            if (i - stillness_frames + 2 - start_frame > 5):
                words.append({
                    "start_frame": start_frame,
                    "end_frame": i - stillness_frames + 2
                })

            next_word_start_index = np.where(np.asarray(diff_arr[i:]) > threshold)[0]
            if len(next_word_start_index):
                start_frame = next_word_start_index[0] + i
                i = start_frame + 1
                still_count = 0
            else:
                break
        else:
            i += 1

    return words


# Preprocess
# ----------
def align_face_img(img):
    img = np.asarray(img)
    img = image_resize(img, width=460)

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

        img_x_middle = img.shape[1] // 2
        img_y_middle = img.shape[0] // 2
        img = Image.fromarray(img)
        img_rotated = img.rotate(dg_angle)

        mouth_x, mouth_y, mouth_w, mouth_h = get_mouth_dlib(shape)
        rotated_x, rotated_y = rotate_point((img_x_middle, img_y_middle), (mouth_x, mouth_y), angle)

        rotated_x = int(rotated_x)
        rotated_y = int(rotated_y)

        y_pad = (76 - mouth_h) // 2
        x_pad = (76 - mouth_w) // 2

        cropped_img = np.array(img_rotated)[rotated_y - y_pad - (mouth_h // 2):rotated_y + (mouth_h // 2) + y_pad,
                      rotated_x - x_pad:rotated_x + mouth_w + x_pad]
        cropped_img = cv2.resize(cropped_img, (96, 96))

        return (img_rotated, Image.fromarray(cropped_img))
    else:
        return False


def extract_frames(video_frames):
    frames2return = []

    # with mp.Pool(29) as pool:
    #     results = [pool.apply_async(align_face_img, (f,)) for f in video_frames]
    #
    #     for r in results:
    #         isValid = r.get()
    #         if isValid is not False:
    #             rotadet_img, cropped_img = isValid
    #             frames2return.append(cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2GRAY))

    results = [align_face_img(f) for f in video_frames]

    for r in results:
        isValid = r
        if isValid is not False:
            rotadet_img, cropped_img = isValid
            frames2return.append(cv2.cvtColor(np.array(cropped_img), cv2.COLOR_BGR2GRAY))

    print("DONE")
    return frames2return


# Returns (frames, num_padding_frames)
def get_word_frames(video_frames, word):
    start = word["start_frame"]
    end = word["end_frame"]
    print(f"start: {start}, end: {end}")

    word_frames_length = end - start + 1
    print(f"length: {word_frames_length}")

    middle = ((end - start) // 2) + start
    print(f"middle: {middle}")

    padding_frame = np.zeros((96, 96))
    frames_count = 29
    before = 10
    after = 19

    if (word_frames_length == frames_count):  # exact num of frames
        return video_frames[start:end + 1], 0

    elif (word_frames_length < frames_count):  # less than needed frames for word
        if middle < before:  # frame of middle of word is less than what we need from before
            if (len(video_frames) >= frames_count):  # enough frames in video
                return video_frames[:frames_count], 0

            else:  # not enough frames in video
                return video_frames, 29 - len(video_frames)
        else:  # frame of middle of word is what we need from befor or more
            if (len(video_frames) >= middle + after - 1):  # enough frames remaining in video
                return video_frames[middle - before:middle + after], 0
            else:  # not enough frames remaining in video (word at the end)
                return video_frames[middle - before:], after - (len(video_frames) - middle)
    else:  # more than needed frames for word
        return video_frames[:frames_count], 0


def video_process(video_path, words_input):
    if os.path.exists(f"{root_path}/npzs"):
        shutil.rmtree(f"{root_path}/npzs")

    file_name = video_path.split('/')[-1].split('.')[0]
    print(file_name)

    frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is not True:
            break
        frames.append(frame)
    cap.release()

    file_paths = []

    for i in range(0, len(words_input)):
        relevant_frames, num_padding = get_word_frames(frames, words_input[i])

        frames_to_save = extract_frames(relevant_frames)

        # less than needed frames for word
        padding_frame = np.zeros((96, 96))
        if num_padding > 0:
            frames_to_save = np.append(frames_to_save, [padding_frame for _ in range(num_padding)], axis=0)

        # more than needed frames for word
        if len(frames_to_save) > 29:
            frame_to_delete = 2
            while len(frames_to_save) > 29:
                if len(frames_to_save) - 1 > frame_to_delete:
                    frames_to_save = np.delete(frames_to_save, frame_to_delete, axis=0)
                else:
                    frames_to_save = np.delete(frames_to_save, len(frames_to_save) - 1, axis=0)
                frame_to_delete += 2

        npz_path = f"{root_path}/npzs/{i}.npz"
        pathlib.Path('/'.join(npz_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, data=frames_to_save)
        file_paths.append(npz_path)

    return file_paths


def send_files(url, filenames):
    files = []
    for filename in filenames:
        with open(filename, "rb") as file:
            file_data = file.read()
            files.append(("files", (filename, file_data, "multipart/form-data")))

    response = requests.post(url, files=files, verify=False)

    if response.status_code == 200:
        print("Files uploaded successfully.")
        return response.json()
    else:
        print("Error uploading the files.")
        raise Exception


# Run prediction
def run_model():
    video_path_input = f"{root_path}/video.mp4"
    words_input = get_words_from_video(video_path_input)
    file_paths = video_process(video_path_input, words_input)

    prediction = send_files('https://9dfb-34-73-176-7.ngrok.io/predict', file_paths)

    print(prediction)

    return prediction
