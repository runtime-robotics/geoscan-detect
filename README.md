# Geoscan Mini 2: детекция rover

Минимальная схема для ручного полёта: штатная камера Geoscan → ROS 2 → YOLO11n → изображение с рамками и confidence. Скрипты не вызывают arm, takeoff, land и не отправляют команды управления. Пилот подводит дрон к объекту пультом.

Репозиторий содержит инструменты. Реальные снимки, разметку и обученные веса ещё нужно получить. Файла конкурсного задания в исходном репозитории не было; по согласованию выбран один класс `rover`, ID `0`.

## 1. Проверить борт и штатные сервисы

Подключиться к Wi-Fi дрона, открыть Pioneer Code: <http://10.42.0.1:9090/>. Терминал доступен в CodeOSS на порту 9999, SSH: `ssh pioneermini@10.42.0.1`. Команды ниже выполняются в Bash на борту из корня репозитория.

```bash
source /opt/ros/jazzy/setup.bash
ros2 node list
ros2 topic list -t
ros2 service list
systemctl --type=service --state=running
tr '\0' '\n' < /proc/device-tree/compatible
```

Сохранить результат последней команды для экспорта RKNN. Целевой чип нельзя выбирать по названию дрона. Например, для строки `rockchip,rk3588s` используется target `rk3588`.

Не останавливать и не переустанавливать службы Geoscan. Не открывать `/dev/video*` вторым процессом. Использовать уже опубликованный топик основной камеры, не камеры оптического потока. Его имя найти через `ros2 topic list -t` и проверить:

```bash
CAMERA_TOPIC=/имя/реального/топика
ros2 topic info "$CAMERA_TOPIC" --verbose
ros2 topic hz "$CAMERA_TOPIC"
```

Нужен тип `sensor_msgs/msg/Image` или `sensor_msgs/msg/CompressedImage`. Для второго добавить `--compressed` во все вызовы collect/detect. Подписка BEST_EFFORT совместима с обоими режимами надёжности издателя; очередь и обработка последнего кадра ограничивают задержку.

Если ROS-камера отсутствует, сначала проверить конфигурацию установленного образа и `gs_camera`. Официальный полный запуск:

```bash
ros2 launch gs_core pioneer_mini2_full_launch.py
```

Запускать его только если соответствующие ноды ещё не работают: он поднимает и камеру, и узел автопилота. Сам запуск не нужен при уже работающем штатном стеке. На образах с одним RTSP-потоком потребуется включить ROS-камеру штатным способом; эти скрипты принимают ROS изображения, а не RTSP. Названия systemd units зависят от версии образа, поэтому здесь нет выдуманных команд `systemctl restart`.

## 2. Снять и разметить датасет

Для сбора достаточно системных `rclpy`, `sensor_msgs`, `cv_bridge`, OpenCV. Если импорты отсутствуют, установить на Ubuntu борта:

```bash
sudo apt install ros-jazzy-cv-bridge python3-opencv
python3 scripts/collect.py --topic "$CAMERA_TOPIC" --fps 1 --count 300
```

Снимки попадут в отдельную серию `captures/<UTC-время>/`. Перезапуск не затирает старую серию. Ctrl+C завершает сбор. Скрипт без кадров выдаёт ошибку через 15 секунд.

Снять несколько разных серий: rover с высоты предполагаемого полёта, разные направления, фон, свет, расстояния; добавить пустой фон. Удалить смазанные кадры. 300 снимков подходят для первого прогона, но качество зависит от разнообразия и размера rover в кадре. Сбор можно проводить на земле с включённой камерой и затем дополнить реальными пролётами.

Скопировать снимки на рабочий компьютер. В CVAT или другом редакторе создать задачу Object detection с единственной меткой `rover`. Обвести каждый видимый rover прямоугольником, без большого запаса фона. Экспортировать разметку в YOLO Detection. Для негативных кадров оставить пустой `.txt`.

Разделить **целые серии** примерно 70/20/10 на train/val/test. Соседние кадры одного пролёта не должны попасть в разные части. После экспорта разложить файлы так:

```text
dataset/
  images/train/flight1_001.jpg
  images/val/flight2_001.jpg
  images/test/flight3_001.jpg
  labels/train/flight1_001.txt
  labels/val/flight2_001.txt
  labels/test/flight3_001.txt
```

Имена изображения и метки должны совпадать. Строка метки: `0 center_x center_y width height`; координаты нормированы в диапазон 0..1. Все объекты имеют ID 0. При объединении серий сохранять уникальные имена. YAML уже есть в `config/rover.yaml`.

## 3. Обучить на компьютере

Нужен Python 3.11 или 3.12, не системный Python 3.14 CachyOS. Например, с установленным `uv`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/train.py --device 0
# Без NVIDIA: python scripts/train.py --device cpu
```

Обучение стартует с `yolo11n.pt`, размер 640, 100 эпох. В конце скрипт печатает точный путь к best.pt. Повторный запуск Ultralytics создаёт следующий каталог `rover2` и т.д. Проверить на отдельной тестовой серии:

```bash
yolo detect val model=runs/rover/weights/best.pt data=runs/rover-data.yaml split=test
yolo detect predict model=runs/rover/weights/best.pt source=dataset/images/test save=True
```

Посмотреть ошибки и пропуски на сохранённых изображениях. Большой mAP на соседних кадрах не подтверждает работу на новом фоне.

## 4. Экспорт для NPU

### Штатный сервис Geoscan

```bash
python scripts/export.py runs/rover/weights/best.pt --format onnx
```

Загрузить `best.onnx` в раздел «Модели ИИ» на <http://10.42.0.1:7777/>, архитектура `yolov11`. Согласно документации, сервис принимает ONNX с opset 19. Это путь через штатный сервис конвертации и Pioneer-RKNN; сам ONNX не является готовой моделью для NPU.

Наш `detect.py` использует Ultralytics backend и ожидает его собственный RKNN экспорт. Файл RKNN из регистра Pioneer-RKNN может иметь другой формат выходных тензоров, поэтому не подставлять его в этот скрипт без проверки. Отдельную обёртку Pioneer-RKNN здесь не добавляем.

### RKNN для detect.py

На x86_64 Linux с Python 3.12, в том же окружении обучения:

```bash
uv pip install rknn-toolkit2
# rk3588 здесь только пример. Указать чип из /proc/device-tree/compatible борта.
python scripts/export.py runs/rover/weights/best.pt --format rknn --chip rk3588
```

Экспорт выдаёт каталог `best_rknn_model/`, внутри RKNN и `metadata.yaml` с классами. Скопировать **весь каталог** на дрон в `models/best_rknn_model/`. Начальный экспорт без INT8-калибровки. Сначала проверить детекции на NPU; INT8 оптимизация для этой задачи не требуется.

На борту оставить штатный драйвер NPU и библиотеки Geoscan. Для нашего backend нужна `rknnlite` из `rknn-toolkit-lite2`, совместимая с версией драйвера и экспортера. Проверить:

```bash
python3 -c 'from rknnlite.api import RKNNLite; print("RKNNLite available")'
```

Если модуля нет, установить подходящий ARM64 wheel RKNN Toolkit Lite2 от Rockchip в отдельное окружение. Нельзя заменять системный драйвер или ставить x86 wheel на борт. Ошибки версии RKNN решать подбором совместимых toolkit/runtime, не переустановкой ROS.

## 5. Запуск и топик результата

На борту, Python 3.12 из Ubuntu 24.04:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install ultralytics==8.4.14 'numpy<2' 'opencv-python<4.12'
python -c 'import rclpy, cv_bridge; from rknnlite.api import RKNNLite'
python scripts/detect.py --topic "$CAMERA_TOPIC" --model models/best_rknn_model --fps 5
```

`--system-site-packages` нужен для штатных ROS и RKNN модулей. Если появляются ошибки ABI NumPy/cv_bridge, сверить версии NumPy окружения и системного cv_bridge; не обновлять системные библиотеки вслепую.

Результат:

| Топик | Тип | Содержимое |
| --- | --- | --- |
| `/geoscan_detect/image` | `sensor_msgs/msg/Image` | BGR кадр с рамками, классом и confidence |
| `/geoscan_detect/image/compressed` | `sensor_msgs/msg/CompressedImage` | Тот же кадр JPEG для передачи по Wi-Fi |

Timestamp и frame_id сохраняются из камеры. `device='cpu'` у вызова Ultralytics не отключает RKNN: при выборе RKNN-каталога вычисления выполняет RKNNLite на NPU. Для проверки логики на CPU можно скопировать `best.pt` и передать его вместо каталога. Медленный CPU-прогон не подтверждает работу NPU.

В ROS-просмотрщике установленной веб-панели выбрать `/geoscan_detect/image` или JPEG-топик с соответствующим типом. Обновить список топиков после первого кадра. Проверка из терминала:

```bash
ros2 topic info /geoscan_detect/image --verbose
ros2 topic hz /geoscan_detect/image
```

**Ограничение штатной панели:** открытая документация Pioneer Code описывает сервисы, но не гарантирует просмотр произвольного ROS Image в каждой версии. На реальном образе это ещё не проверено. Раздел «Модели ИИ» сам по себе не запускает наш обработчик и не показывает его выходной топик.

Если в панели нет выбора ROS-топика, открыть поток в браузере через стандартный ROS web_video_server, на борту в другом терминале:

```bash
sudo apt install ros-jazzy-web-video-server
source /opt/ros/jazzy/setup.bash
ros2 run web_video_server web_video_server --ros-args -p port:=8080
```

Открыть <http://10.42.0.1:8080/stream?topic=/geoscan_detect/image>. Проверить, что порт 8080 свободен, иначе выбрать другой. Это дополнительная страница просмотра на борту; интеграция внутрь меню Geoscan потребует сведений о конкретной версии панели.

Перед полётом проверить на земле: обычная камера видна, rover получает рамку, пустой фон приходит без рамок, JPEG и raw топики обновляются. Затем проверить перенос на новый фон, загрузку CPU и задержку видео. Скрипт не управляет направлением камеры. Выставить её штатными средствами до ручного пролёта. Остановка: Ctrl+C.

## Проверки и источники

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts tests
```

Локальные тесты не подтверждают совместимость драйвера NPU, камеры и панели конкретного дрона. Снимков и доступа к борту при подготовке не было; обучение и экспорт реальных весов не выполнены.

Документация проверена 18 ноября 2026:

- [Geoscan: ROS 2, gs_core и штатный launch](https://docs.geoscan.ru/pioneer/soft/pioneer-os/ros2/)
- [Pioneer Code и адреса сервисов](https://docs.geoscan.ru/pioneer/soft/pioneer-os/pioneer-code/)
- [Модели ИИ: ONNX opset 19 и версии библиотек](https://docs.geoscan.ru/pioneer/soft/pioneer-os/pioneer-code/model-ai/)
- [Pioneer-RKNN Yolo](https://docs.geoscan.ru/pioneer/soft/pioneer-os/pio-rknn/Yolo/)
- [Ultralytics RKNN](https://docs.ultralytics.com/integrations/rockchip-rknn/)
- [Rockchip RKNN Toolkit2 и Lite2](https://github.com/airockchip/rknn-toolkit2)
