import numpy as nmp

step_count = 10


def calc_rect(integral_image: nmp.ndarray, lx, ly, rx, ry) -> int:
    a = integral_image[lx, ly]
    b = integral_image[lx, ry]
    c = integral_image[rx, ry]
    d = integral_image[rx, ly]
    return c + a - b - d


def get_ratio_range(ratio_bias, ration_deviation):
    return [
        ratio_bias - ration_deviation,
        ratio_bias - ration_deviation / 2,
        ratio_bias,
        ratio_bias + ration_deviation / 2,
        ratio_bias + ration_deviation
    ]


def generate_horizontal_rects(
        lx, ly, rx, ry,
        min_width,
        min_height,
        ratio_bias,
        ration_deviation
):
    width = rx - lx
    width_step = (width - min_width) // step_count
    height = ry - ly
    height_step = (height - min_height) // step_count
    ratio_range = get_ratio_range(ratio_bias, ration_deviation)
    res = []
    for i in range(step_count + 1):
        f_width = min_width + i * width_step
        for j in range(step_count + 1):
            f_height = min_height + j * height_step
            for fx in range(lx, rx - f_width):
                for fy in range(ly, ry - f_height):
                    for ratio in ratio_range:
                        res.append(HorizontalRectFeature(
                            fx, fy, fx + f_width, fy + f_height, ratio
                        ))
    return res


class Feature(object):

    def calc(self, image) -> int:
        pass


class HorizontalRectFeature(Feature):

    def __init__(self, lx, ly, rx, ry, ratio) -> None:
        super().__init__()
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry
        self.height = int((ry - ly) / (ratio + 1))

    def calc(self, image: nmp.ndarray):
        top = calc_rect(image, self.lx, self.ly, self.rx, self.ry - self.height)
        bottom = calc_rect(image, self.lx, self.ly + self.height, self.rx, self.ry)
        return top - bottom


class Vertical3RectFeature(Feature):

    def __init__(self, lx, ly, rx, ry, ratio) -> None:
        super().__init__()
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry
        self.width = int((ry - ly) / (ratio + 1) / 2)

    def calc(self, image: nmp.ndarray):
        left = calc_rect(image, self.lx, self.ly, self.lx + self.width, self.ry)
        right = calc_rect(image, self.rx - self.width, self.ly, self.rx, self.ry)
        center = calc_rect(image, self.lx + self.width, self.ly, self.rx - self.width, self.ry)
        return left + right - center
