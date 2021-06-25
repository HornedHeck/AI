import random

from features import HorizontalRectFeature, Vertical3RectFeature, Feature
import numpy as nmp
from viola_jones import learn as learn_features
from dataset_utils import get_data
from dataset_utils import to_integral
from classifier_ada import Classifier, learn
from visualizer import show_prediction

eyes_features_params = nmp.array([
    [0.2258, 130, 54, 145, 99, 28, 1, 150],
    [0.6149, 147, 48, 162, 93, 27, -1, 0],
    [0.2246, 148, 45, 163, 99, 40, -1, 25],
    [0.3042, 143, 45, 160, 90, 31, 1, 100],
    [0.2767, 130, 45, 161, 99, 32, -1, -150],
    [0.2380, 39, 50, 54, 95, 34, -1, -150],
    [0.2281, 42, 53, 57, 98, 30, 1, 150],
    [0.5934, 58, 45, 73, 90, 34, -1, -150],
    [0.2836, 53, 51, 68, 98, 27, -1, -75],
    [0.3519, 40, 53, 59, 98, 30, -1, 150]
])

r_eye_features_params = nmp.array([
    [0.2500, 130, 59, 145, 119, 31, 1, -50],
    [0.2308, 148, 69, 163, 129, 31, 1, -100],
    [0.6321, 150, 45, 165, 107, 34, 1, -150],
    [0.2866, 143, 59, 162, 125, 35, 1, -150],
    [0.2527, 134, 55, 155, 127, 33, 1, -50],
    [0.2569, 130, 56, 155, 130, 41, -1, -150],
    [0.2756, 136, 53, 163, 123, 37, 1, -50],
    [0.2662, 132, 53, 161, 115, 34, -1, 0],
    [0.2215, 130, 49, 161, 127, 40, -1, -150],
])
l_eye_features_params = nmp.array([
    [0.5653, 59, 47, 74, 109, 31, -1, -150],
    [0.2397, 35, 53, 50, 117, 34, 1, 150],
    [0.2832, 57, 57, 72, 125, 35, -1, -150],
    [0.2206, 45, 51, 60, 121, 31, -1, -150],
    [0.2895, 47, 45, 62, 119, 41, -1, -150],
    [0.1913, 51, 69, 68, 129, 33, 1, -150],
    [0.1707, 37, 51, 54, 117, 33, 1, -150],
    [0.3078, 39, 49, 58, 111, 34, 1, 100],
    [0.4182, 48, 69, 69, 129, 33, 1, 50],
    [0.2464, 39, 57, 68, 117, 30, 1, 50],
])

alt_eyes_features_params = nmp.array([
    [0.2668, 35, 54, 50, 114, 32, 1, 150],
    [0.2626, 38, 45, 53, 105, 33, 1, 150],
    [0.4939, 59, 48, 74, 108, 30, -1, -150],
    [0.3013, 47, 45, 62, 119, 41, 1, -150],
    [0.4584, 48, 69, 69, 129, 33, 1, 25],
    [0.3208, 132, 59, 147, 119, 33, 1, 125],
    [0.3337, 148, 69, 163, 129, 31, 1, -100],
    [0.5981, 149, 45, 164, 105, 33, 1, -50],
    [0.2447, 133, 47, 152, 111, 33, -1, -50],
    [0.3064, 142, 58, 163, 126, 35, -1, -150]
])

face_features_params = nmp.array(
    [
        [0.4876, 53, 50, 123, 90, 20, -1, -150],
        [0.2892, 58, 77, 128, 117, 22, -1, 75],
        [0.2683, 77, 64, 147, 107, 24, 1, -75],
        [0.3331, 50, 57, 120, 115, 30, -1, 75],
        [0.3071, 68, 55, 141, 95, 20, 1, 50],
        [0.4167, 64, 65, 143, 105, 20, -1, 100],
        [0.2804, 67, 50, 146, 108, 33, -1, 150],
        [0.2903, 55, 51, 146, 91, 21, -1, -150],
        [0.2183, 50, 67, 147, 107, 21, -1, 100]
    ]
)

mouth_1_fp = nmp.array([
    [0.3250, 65, 95, 115, 133, 20, 1, 150],
    [0.2842, 58, 84, 118, 114, 17, -1, 100],
    [0.3284, 59, 120, 124, 150, 15, 1, 100],
    [0.7234, 56, 104, 121, 146, 25, 1, 150],
    [0.2763, 53, 84, 118, 134, 37, 1, 100],
    [0.3693, 64, 104, 139, 134, 22, -1, -50],
    [0.3780, 70, 98, 145, 136, 24, -1, -150],
    [0.2686, 54, 108, 134, 138, 19, 1, 50],
    [0.3601, 50, 95, 130, 133, 20, 1, 150],
    [0.4311, 54, 88, 139, 118, 20, -1, 50],
])

mouth_2_feature_params = nmp.array([
    [0.3576, 62, 120, 112, 150, 11, -1, -150],
    [0.2190, 64, 111, 114, 143, 12, -1, -150],
    [0.2889, 80, 111, 130, 145, 13, 1, -150],
    [0.2075, 60, 110, 110, 145, 13, 1, -150],
    [0.5815, 62, 116, 114, 150, 13, -1, -150],
    [0.2821, 60, 111, 112, 150, 13, -1, -100],
    [0.2551, 72, 110, 130, 147, 14, -1, -150],
    [0.2870, 61, 110, 119, 150, 15, -1, -150],
    [0.3424, 70, 111, 130, 150, 15, -1, -50],
])

nose_fp = nmp.array([
    [0.1063, 60, 81, 110, 106, 9, 1, -100],
    [0.1642, 60, 82, 110, 115, 12, -1, -150],
    [0.1465, 78, 80, 128, 114, 12, -1, -50],
    [0.3465, 80, 86, 130, 120, 12, 1, -150],
    [0.1887, 70, 81, 120, 116, 12, -1, 150],
    [0.1191, 78, 84, 128, 119, 12, -1, -150],
])

# 3 V3 7 H
mouth_fp = nmp.array([
    [0.2748, 72, 110, 122, 140, 10, 1, -150],
    [0.4208, 60, 113, 114, 150, 14, -1, 150],
    [0.4693, 69, 110, 125, 150, 14, -1, 50],
    [0.3424, 70, 86, 120, 138, 33, 1, 150],
    [0.2758, 90, 90, 140, 142, 31, -1, -150],
    [0.3318, 100, 98, 150, 150, 31, 1, 150],
    [0.7043, 56, 104, 121, 147, 22, -1, 150],
    [0.3001, 50, 82, 115, 134, 38, 1, 100],
    [0.2829, 66, 96, 141, 142, 31, -1, -150],
    [0.3459, 50, 80, 135, 123, 22, -1, 150],
])

face_2_fp = nmp.array([
    [0.2130, 77, 42, 112, 142, 44, -1, -150],
    [0.5180, 82, 25, 117, 131, 38, 1, 125],
    [0.2484, 89, 42, 124, 148, 47, -1, 150],
    [0.2512, 88, 25, 123, 141, 46, -1, -25],
    [0.2932, 81, 25, 116, 143, 46, -1, -150],
    [0.2055, 82, 48, 118, 148, 36, 1, -125],
    [0.1874, 85, 37, 121, 137, 36, 1, -125],
    [0.4380, 76, 49, 116, 149, 37, 1, -100],
    [0.2498, 78, 49, 122, 149, 36, 1, -150],
])


def generate_hr_feature(params: iter) -> HorizontalRectFeature:
    f = HorizontalRectFeature(*params[1:6])
    f.s = params[6]
    f.theta = params[7]
    return f


def generate_v3r_feature(params: iter) -> Vertical3RectFeature:
    f = Vertical3RectFeature(*params[1:6])
    f.s = params[6]
    f.theta = params[7]
    return f


def hr_from_params(params: nmp.ndarray):
    return Classifier(
        params[:, 0],
        list([generate_hr_feature(p) for p in params])
    )


def v3r_from_params(params: nmp.ndarray):
    return Classifier(
        params[:, 0],
        list([generate_v3r_feature(p) for p in params])
    )


def generate_strong_c():
    strong_a = nmp.array([
        0.5653075098771412,
        -0.2682554731089583,
        0.2657941372784275,
        0.08666735243198694,
        0.12632524688731525,
        0.5433065829457234,
        -0.15719967921847625,
        4.440892098500624e-16,
        0.11829996644475804,
        0.26939489353451224
    ])
    strong_c = [
        hr_from_params(eyes_features_params),
        hr_from_params(r_eye_features_params),
        hr_from_params(l_eye_features_params),
        hr_from_params(alt_eyes_features_params),
        hr_from_params(face_features_params),
        hr_from_params(mouth_1_fp),
        v3r_from_params(mouth_2_feature_params),
        v3r_from_params(nose_fp),
        Classifier(
            mouth_fp[:, 0],
            list([generate_v3r_feature(p) for p in mouth_fp[:3]]) + list([generate_hr_feature(p) for p in mouth_fp[3:]])
        ),
        hr_from_params(face_2_fp)
    ]
    return strong_a, strong_c


def calc_strong(a: nmp.ndarray, c: list[Classifier], image: nmp.ndarray):
    res = 0.
    for a_i, c_i in zip(a, c):
        res += a_i * c_i.calc(image)
    return nmp.sign(res)


def run_strong_test():
    a_s, c_s = generate_strong_c()
    x, y = get_data()

    correct = 0
    for i in range(len(y)):
        x_int = to_integral(x[i])
        prediction = calc_strong(a_s, c_s, x_int)
        if prediction == y[i]:
            correct += 1

    print(f'{correct} / {len(y)}')


def run_strong_showcase(idx):
    a_s, c_s = generate_strong_c()
    x, y = get_data()
    for i in idx:
        x_int = to_integral(x[i])
        show_prediction(x[i], y[i], calc_strong(a_s, c_s, x_int))


def get_classifiers():
    eyes_c = hr_from_params(eyes_features_params)
    r_e_c = hr_from_params(r_eye_features_params)
    l_e_c = hr_from_params(l_eye_features_params)
    alt_eyes_c = hr_from_params(alt_eyes_features_params)
    face_c = hr_from_params(face_features_params)
    mouth_1_c = hr_from_params(mouth_1_fp)
    mouth_2_c = v3r_from_params(mouth_2_feature_params)
    nose_c = v3r_from_params(nose_fp)
    mouth_c = Classifier(
        mouth_fp[:, 0],
        list([generate_v3r_feature(p) for p in mouth_fp[:3]]) + list([generate_hr_feature(p) for p in mouth_fp[3:]])
    )
    face_2_c = hr_from_params(face_2_fp)
    return [
        eyes_c,
        r_e_c,
        l_e_c,
        alt_eyes_c,
        face_c,
        mouth_1_c,
        mouth_2_c,
        nose_c,
        mouth_c,
        face_2_c
    ]


a_global = nmp.array([
    0.09314362065766518, 0.046289986331188325, 0.13057133829838477, 0.14099124393182852, 0.4401373411937069,
    0.11518886610612925, 0.009273063739395425, 0.12427907147723613, 0.11746510240259343, 0.0433249672050389,
    0.1702141295420171, 0.28161970515273066, 0.020597801948540592, 0.21517770653201682, 0.06152451392640002,
    0.06047441784683676, 0.05373953261935428, 0.1656842703269554, 0.19439952406748648, 0.1895102670677505,
    0.1203322423323726, 0.6321397052784632, 0.078268326769534, 0.06666718076467666, 0.24250414196994283,
    0.2906294784973147, 0.044154724795747766, 0.27441097319963165, 0.025680554304588467, 0.20321445292257007,
    0.3138268296372748, 0.3892147434258533, 0.01516886824067157, 0.10788490181136073, 0.16846541453575176,
    0.12324116180987196, 0.2811644639460983, 0.21477314232067396, 0.20918743083443936, 0.05825069345640815,
    0.12283688423258499, 0.27136269903602594, 0.052412359723350024, 0.1701310491344636, 0.03268884999485706,
    0.03990488661390714, 0.18260924053091418, 0.2595452668896482, 0.044768607695173816, 0.2967500901625248,
    0.3289896206839641, 0.0408477766727357, 0.04656978970143212, 0.0799455490264698, 0.05617210747932055,
    0.21027262784812148, 0.23194618900495836, 0.20815556676737068, 0.27537874705576054, 0.19230083916982074,
    0.17362119095377018, 0.16072887643490794, 0.014146426587569698, 0.004691192586735229, 0.001133170640692501,
    2.2204460492503126e-16, 2.2204460492503126e-16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15582944206563887,
    0.09871641366657286, 0.1534232327949691, 0.0032977325451061493, 0.6151776412869627, 0.05167905831046334,
    0.2203587216356864, 0.12731267818408543, 0.1620472127034639, 0.04157417969889672, 0.041494777398287545,
    0.2915564701525505, 0.37181075208427655, 0.07913906852837312, 0.15335127217031982, 0.19538761254278128
])


def test_global_classifier():
    cls = build_global_classifier()
    x, y = get_data()

    correct = 0
    for i in range(len(y)):
        x_int = to_integral(x[i])
        prediction = cls.calc(x_int)
        if prediction == y[i]:
            correct += 1
        else:
            show_prediction(x[i] , y[i], prediction)

    print(f'{correct} / {len(y)}')


def build_global_classifier():
    cls = get_classifiers()
    features = []
    for c in cls:
        features += c.features
    cls = Classifier(
        a_global,
        features
    )
    return cls


def learn_global_classifier():
    features = build_global_classifier().features
    learn_features(features, len(features))


if __name__ == '__main__':
    test_global_classifier()
# run_strong_test()
# pics = nmp.random.randint(0, 160, 20)
# run_strong_showcase(pics)
# classifiers = get_classifiers()
# c_len = len(classifiers)
# predictions = nmp.zeros(c_len)
#
# learn(classifiers, c_len)

# x, y = get_data()
# correct = nmp.zeros(c_len)
# correct = 0
# for i in range(20):
#     j = random.randint(0, len(y) - 1)
# j = i
# x_int = to_integral(x[j])
# prediction = calc_strong(classifiers, x_int)
# if prediction == y[j]:
#     correct += 1
# for k in range(c_len):
#     predictions[k] = classifiers[k].calc(x_int)
#     if predictions[k] == y[j]:
#         correct[k] += 1
# show_prediction(x[j], y[j], predictions)

# print(f'{correct} / {len(y)}')
