import numpy as nmp
import torch
from matplotlib import pyplot as plt

from flowers.cnn import FlowersCNN
from flowers.dataset import UKFlowers, Type

_NAMES = [
    "розовая примула", "жестколистная карманная орхидея", "кентерберийские колокольчики",
    "душистый горошек", "бархатцы английские", "тигровая лилия", "лунная орхидея",
    "райская птица", "монашество", "чертополох", "львиный зев",
    "лапка жеребенка", "король протея", "чертополох", "желтый ирис",
    "глобус", "пурпурный эхинококк", "перуанская лилия", "воздушный шар",
    "гигантская белая арум лилия", "огненная лилия", "игольница", "рябчик",
    "красный имбирь", "виноградный гиацинт", "кукурузный мак", "перья принца Уэльского",
    "горечавка без стебля", "артишок", "сладкий вильям", "гвоздика",
    "садовые флоксы", "любовь в тумане", "мексиканская астра", "альпийский падуб",
    "рубиногубая каттлея", "плащ-цветок", "мастурбар", "сиамский тюльпан",
    "постная роза", "барбетонная маргаритка", "нарцисс", "лилия-меч", "пуансеттия",
    "болеро темно-синее", "желто-белый", "календула", "лютик", "бычья маргаритка",
    "одуванчик обыкновенный", "петуния", "анютины глазки", "примула", "подсолнечник",
    "пеларгония", "епископ Лландаффа", "гаура", "герань", "оранжевая георгина",
    "розово-желтый георгин?", "cautleya spicata", "анемон японский",
    "черноглазая сьюзан", "сильвербуш", "калифорнийский мак", "остеоспермум",
    "весенний крокус", "ирис бородатый", "ветроцвет", "мак древесный", "газания",
    "азалия", "водяная лилия", "роза", "шиповник", "ипомея",
    "пассифлора", "лотос", "жаба лилия", "антуриум", "франжипани",
    "клематис", "гибискус", "коломбина", "роза пустыни", "мальва древовидная",
    "магнолия", "цикламен", "кресс-салат", "канна лилия", "гиппеаструм",
    "пчелиный бальзам", "шар мох", "наперстянка", "бугенвиллея", "камелия", "мальва",
    "мексиканская петуния", "бромелия", "покрывной цветок", "труба лиана",
    "ежевичная лилия"
]

model_path = '''/home/hornedheck/PycharmProjects/AI/models/flowers_cnn_99.252_23_59_31.pt'''

model = FlowersCNN()
model.load_state_dict(torch.load(model_path))

dataset = UKFlowers('''/home/hornedheck/PycharmProjects/AI/dataset/flowers''', Type.VAL, False)
with torch.no_grad():
    count = len(dataset)
    items = nmp.random.randint(0, count, (10, 2), dtype=int)
    for i in items:
        x, y = dataset[i]
        x_t = torch.tensor(x)
        res = model(x_t).argmax(1)

        plt.imshow(x[0])
        plt.title(f'Got: {_NAMES[res[0].item()]}, Expected: {_NAMES[y[0]]}')
        plt.show()
        plt.imshow(x[1])
        plt.title(f'Got: {_NAMES[res[1].item()]}, Expected: {_NAMES[y[1]]}')
        plt.show()