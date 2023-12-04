# Вычисление оптического потока

## Описание задачи

Язык программирования - Python.
Разрешено использовать любую библиотеку для машинного обучения (PyTorch, TensorFlow, Keras и др.). Запрещено
использование готовых архитектур "из коробки" одной строчкой, в коде должны быть прописаны слои модели.

Необходимо создать модель для вычисления оптического потока изображения, обучить ее на предложенном датасете, а также
измерить время работы модели на тестовом датасете и посчитать метрику.

## Датасет

Датасет содержит 1485 примеров, каждый из которых состоит из 2 изображений и соответсвующего им оптического потока (<
sample_id>_img1.ppm <sample_id>_img2.ppm <sample_id>_flow.flo).<br/>
[Ссылка на датасет](https://drive.google.com/file/d/1ipbk2nGpVlTHHY7i4GfTf-cJ0VHfvURd/view?usp=sharing).

Предложенный датасет представляет из себя выборку из синтетического датасета 'Flying Chairs'. Изображения показывают
рендеры 3D-моделей кресел, движущихся перед случайным фоном, при этом движения как стульев и фона являются чисто
плоскостными.

```
@InProceedings{DFIB15,
  author    = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz{\i}rba{\c{s}} and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
  title     = "FlowNet: Learning Optical Flow with Convolutional Networks",
  booktitle = "IEEE International Conference on Computer Vision (ICCV)",
  month     = " ",
  year      = "2015",
  url       = "http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15"
}
```

Для загрузки можно использовать функции, представленные в файлах [utils.py](./utils.py) и [loader.py](./loader.py).

## Метрики

Для оценки качества модели используется следующая метрика:

* EPE - End-point error. <br/>

Метрика рассчитывается как Евклидово расстояние между истинным оптическим потоком и полученным в результате вычислений.

```math
 EPE = ||V_{gt} - V_{calc}||_2 = \sqrt{(\Delta x_{gt} - \Delta x_{calc})^2 + (\Delta y_{gt} - \Delta y_{calc})^2}
```

Реализация расчета метрики на python с использованием pytorch представлена в файле [utils.py](./utils.py) в функции
_mean_epe_

## Baseline

Модель FlowNet S, оптимизатор Adam, loss - EPE (совпадает с метрикой), batch_size = 8, количество эпох 20.

Результаты:

| EPE (train-val set) | EPE (test set) | Time per image, sec |
|---------------------|----------------|---------------------|
| 8.04                | 7.0            | 0.0107              |

Реализация расчета метрики на python с использованием pytorch представлена в файле [utils.py](./utils.py) в функции
_mean_epe_

## Команды для запуска

Помощь по флагам

```bash
 python .\main.py --help
```

Обучение и тест

```bash
 python main.py 'test_dataset_path' --split-value 0 -e --pretrained --load-path 'path' --load-best
```

Запуск на тестовой выборке с наилучший чекпоинтом по EPE

```bash
 python main.py 'dataset_path' -b16 --solver adam --epochs 100 --seed 42
```

Просмотр статистики (необходимо наличие папки processed с чекпоинтами внутри)

```bash
 tensorboard --logdir=./processed   
```

## Результаты

Модель FlowNetS, оптимизатор Adam, loss - EPE (совпадает с метрикой), batch_size = 16, количество эпох 100.

Результаты:

| EPE (train set) | EPE (test set) | Time per image, sec |
|-----------------|----------------|---------------------|
| 4.493           | 7.047          | 0.0183              |


[График метрики](https://github.com/Simraki/CV_2023_lab_6/blob/master/images/plot.png)