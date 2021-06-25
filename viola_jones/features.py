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
        ratio_bias - ration_deviation / 4 * 3,
        ratio_bias - ration_deviation / 2,
        ratio_bias - ration_deviation / 4,
        ratio_bias,
        ratio_bias + ration_deviation / 4,
        ratio_bias + ration_deviation / 2,
        ratio_bias + ration_deviation / 4 * 3,
        ratio_bias + ration_deviation
    ]


def generate_horizontal_rects(
        lx, ly, rx, ry,
        min_width,
        min_height,
        ratio_bias,
        ration_deviation,
):
    width = rx - lx
    width_step = (width - min_width) // step_count
    height = ry - ly
    height_step = (height - min_height) // step_count
    ratio_range = get_ratio_range(ratio_bias, ration_deviation)
    res = []
    thetas = range(-150, 151, 50)
    for i in range(step_count + 1):
        f_width = min_width + i * width_step
        for j in range(step_count + 1):
            f_height = min_height + j * height_step
            x_step = (rx - lx - f_width) // step_count
            if x_step == 0:
                x_step = 1
            y_step = (ry - ly - f_height) // step_count
            if y_step == 0:
                y_step = 1
            for fx in range(lx, rx - f_width + 1, x_step):
                for fy in range(ly, ry - f_height + 1, y_step):
                    for ratio in ratio_range:
                        for theta in thetas:
                            f = HorizontalRectFeature(
                                fx, fy, fx + f_width, fy + f_height, ratio
                            )
                            f.theta = theta
                            res.append(f)

    return res


def generate_test(
        lx, ly, rx, ry,
        min_width,
        min_height,
        ratio_bias,
        ration_deviation,
        count: int = 100
):
    width = rx - lx
    width_step = (width - min_width) // step_count
    height = ry - ly
    height_step = (height - min_height) // step_count
    res = []
    for i in range(step_count + 1):
        f_width = min_width + i * width_step
        for j in range(step_count + 1):
            f_height = min_height + j * height_step
            for fx in range(lx, rx - f_width, 3):
                for fy in range(ly, ry - f_height, 3):
                    res.append(HorizontalRectFeature(
                        fx, fy, fx + f_width, fy + f_height, ratio_bias
                    ))
                    if len(res) >= count:
                        return res
    return res


def generate_vertical_3rects(
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
    thetas = range(-150, 151, 50)
    for i in range(step_count + 1):
        f_width = min_width + i * width_step
        for j in range(step_count + 1):
            f_height = min_height + j * height_step
            x_step = (rx - lx - f_width) // step_count
            if x_step == 0:
                x_step = 1
            y_step = (ry - ly - f_height) // step_count
            if y_step == 0:
                y_step = 1
            for fx in range(lx, rx - f_width + 1, x_step):
                for fy in range(ly, ry - f_height + 1, y_step):
                    for ratio in ratio_range:
                        for theta in thetas:
                            f = Vertical3RectFeature(
                                fx, fy, fx + f_width, fy + f_height, ratio
                            )
                            f.theta = theta
                            res.append(f)
    return res


class Feature(object):

    def __init__(self, s: int = 1, theta: int = 0) -> None:
        super().__init__()
        self.s = s
        self.theta = theta

    def calc(self, image) -> int:
        pass

    def calc_feature(self, image: nmp.ndarray) -> int:
        if self.calc(image) < self.theta:
            return -self.s
        else:
            return self.s


class HorizontalRectFeature(Feature):

    def __init__(self, lx, ly, rx, ry, ratio) -> None:
        super().__init__(s=-1)
        self.lx = int(lx)
        self.ly = int(ly)
        self.rx = int(rx)
        self.ry = int(ry)
        self.height = int((ry - ly) / (ratio + 1))

    def calc(self, image: nmp.ndarray):
        top = calc_rect(image, self.lx, self.ly, self.rx, self.ry - self.height)
        bottom = calc_rect(image, self.lx, self.ly + self.height, self.rx, self.ry)
        return top - bottom

    def __str__(self) -> str:
        return f'HR: [{self.lx}, {self.ly}, {self.rx}, {self.ry}, {self.height}, {self.s}, {self.theta}]'


class Vertical3RectFeature(Feature):

    def __init__(self, lx, ly, rx, ry, ratio) -> None:
        super().__init__()
        self.lx = int(lx)
        self.ly = int(ly)
        self.rx = int(rx)
        self.ry = int(ry)
        self.width = int((ry - ly) / (ratio + 1) / 2)

    def calc(self, image: nmp.ndarray):
        left = calc_rect(image, self.lx, self.ly, self.lx + self.width, self.ry)
        right = calc_rect(image, self.rx - self.width, self.ly, self.rx, self.ry)
        center = calc_rect(image, self.lx + self.width, self.ly, self.rx - self.width, self.ry)
        return left + right - center

    def __str__(self) -> str:
        return f'V3R: [{self.lx}, {self.ly}, {self.rx}, {self.ry}, {self.width}, {self.s}, {self.theta}]'
