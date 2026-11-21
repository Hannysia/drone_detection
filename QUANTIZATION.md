Фаза A — CPU-квантизація, усунення P0-1, ablation по етапах та роздільностях (1280x1280, 640x640)

A1. Оточення, залежності та конфіг (requirements-local.txt, pyproject.toml, src/config.py, tests/test_config_paths.py): зафіксувати CPU quantization stack (onnx, onnxslim, onnxruntime, openvino, nncf, packaging), прибрати застарілі посилання на yolo26n_final_model_best.* і .engine, задати шляхи під yolo26n_run_with_negatives_best.pt та ONNX/OpenVINO моделі для 1280 і 640.

A2. Безпорогова CPU-оцінка (src/processing/evaluation.py): додати явний параметр device у evaluate_model та підтримку директорій *_openvino_model/ замість суворого is_file(); для квантизації використовувати суто evaluate_model(..., rect=False) і evaluate_threshold_free_slices.


A3. Ядро експорту та PTQ-квантизації (P0-1) (src/processing/quantization.py): preflight HF через assert_hf_ready, стратифікований train-only calibration subset (без витоку val/test), ізольований по роздільностях (1280 і 640) покроковий pipeline PyTorch FP32 → ONNX FP32 → ONNX INT8 (кастомний quantize_static: QuantFormat.QDQ, reduce_range=False для сумісності з ARM64 Raspberry Pi і x86_64, виключення вузлів end2end-голови /model.23/ DFL/Sigmoid/TopK) та OpenVINO FP32 → OpenVINO INT8 (nncf, QuantizationPreset.MIXED), підрахунок розміру файлу й директорії.

A4. CPU-бенчмарк, метрики, звіти та HF-артефакти (src/processing/quantization.py): бенчмарк на реальних зображеннях у режимах compute-only та end-to-end (p50/p95/p99, FPS, RAM/CPU), повне безпорогове оцінювання кожного етапу і формату на test (з фіксованим baseline-порогами без пошуку оптимального детекторного conf), збереження CSV/JSON-звітів, тегований upload у HF з quantization_metadata.json.

A5. CLI-скрипт і тести (scripts/06_quantize_cpu.py — новий, tests/test_quantization.py, tests/test_evaluation.py): винесення запуску квантизації та ablation у скрипт, модульні тести без GPU/мережі.

A6. Локальний прогін і аналіз у ноутбуці (notebooks/08_model_quantization.ipynb): запуск квантизації на локалі, рефакторинг ноутбука суто під візуалізацію CSV/JSON-результатів, таблиці деградації й прискорення між форматами/роздільностями.

Фаза B — Адаптивний ROI-трекінг (1280 Search ↔ 640 Track), відео-ablation і HF Space демо

B1. Хронологічні відеодані baseline/v1 і P0-3 (manifests/baseline/v1/main.csv, trackeval_dataset.py, tracking.py): послідовна подача всіх збережених кадрів за video_id з parse_frame_name з урахуванням пропусків від видалених дублікатів + виправлення GT MOT-датасету (P0-3).
B2. Двомодельний ROI-пайплайн (новий модуль + тести): стан SEARCH (1280x1280 INT8) ↔ TRACK (640x640 INT8 crop), центрування навколо останнього боксу vs прогнозу фільтра Калмана з ByteTrack (коректний крок без множення швидкості на пропущені фрейми), гістерезис втрати цілі (max_lost_frames = 3–5) для одиночної цілі.
B3. Відео-ablation і візуалізація (notebooks/09_mot_analysis.ipynb або окремий звіт): підбір параметрів трекера на val, порівняння Full 1280 vs Resize 640 vs ROI Static vs ROI Kalman на test (FPS, MOT/TrackEval, recall по малих цілях, стабільність треку).
B4. Демо на Hugging Face Space (новий app-модуль): інференс на OpenVINO INT8 (1280/640) + адаптивний ROI + ByteTrack, завантаження відео користувачем → рендер боксів, треку й метрик швидкодії.