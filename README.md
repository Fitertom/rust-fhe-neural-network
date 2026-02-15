# 🔒 PPNN (Privacy-Preserving Neural Network)

**A fully homomorphic neural network built from scratch in Rust.**

<div align="center">

<!-- 
    VIDEO: Upload your demo video (mp4) to the GitHub issue editor or drag-and-drop it here when editing the README on GitHub. 
    GitHub will generate a link like: https://github.com/user/repo/assets/...
-->
<video src="https://github.com/user-attachments/assets/PLACEHOLDER_VIDEO_URL" controls="controls" style="max-width: 700px;">
</video>

<br/>

<!-- 
    ARCHITECTURE: Place your diagram image in a folder like `assets/` and link it here.
-->
<img src="assets/architecture_diagram.png" alt="PPNN Architecture" width="700">

</div>

---

**PPNN** — это полная implementation (с нуля, на Rust) системы распознавания рукописных цифр, работающая поверх гомоморфного шифрования (FHE).

Сервер **никогда** не видит цифру, которую вы нарисовали. Он получает зашифрованные пиксели, производит вычисления (матричное умножение + активация через bootstrapping) и возвращает зашифрованный результат. Только вы (клиент в браузере) можете расшифровать ответ.

---

## 🏗 Архитектура и Безопасность

Проект реализует схему **TFHE (Torus FHE)** с программируемым бутстрэппингом.

### Особенности реализации
В данной демонстрационной версии сервер технически **имеет доступ** к приватному ключу (он генерируется при старте сервера), однако **все вычисления (инференс)** над данными пользователя производятся строго в зашифрованном виде с использованием гомоморфных операций (сложение, умножение на константу, bootstrapping). 

Сервер не "подглядывает" в данные во время вычислений, а выполняет математические операции над "шумом", который превращается в осмысленный ответ только после расшифровки на клиенте.

### Компоненты
1.  **LWE Crypto Engine**: Кастомная реализация на Rust (`src/lwe.rs`, `src/bootstrap.rs`).
2.  **Neural Network**: Двухслойная сеть, обученная на `f64` и квантованная в `u64`.
3.  **Collector**: Инструмент для создания собственного датасета (`static/collector.html`).

---

## 🚀 Запуск и Использование

Проект полностью готов к работе "из коробки" — веса модели уже лежат в репозитории (`model_weights.json`).

### 1. Быстрый старт (Инференс)
Запустите сервер с готовой моделью:
```bash
cargo run --release -- serve
```
Откройте `http://localhost:3000` в браузере. Рисуйте цифры, и сервер будет их угадывать в зашифрованном виде.

### 2. Сбор собственного датасета
Хотите, чтобы сеть узнавала именно ваш почерк?
1. Откройте файл `static/collector.html` в любом браузере (просто перетащите файл или через Live Server).
2. Рисуйте цифры в квадрате.
3. Нажимайте цифру на клавиатуре (0-9), чтобы сохранить семпл.
4. Данные сохраняются в папку `static/my_assets/`.

### 3. Обучение на своих данных
После сбора датасета запустите специальный режим обучения:
```bash
cargo run --release -- train_only_my
```
Это перетренирует модель **только** на ваших собранных примерах (очень быстро) и перезапишет `model_weights.json`. После этого перезапустите сервер (`serve`), чтобы применить новую модель.

### 4. Полное обучение
Если нужно переобучить сеть с нуля на комбинации системных шрифтов и ваших данных:
```bash
cargo run --release -- train
```

---

## 🛠 Технические детали (Solutions & Fixes)

*   **Rust**: Core logic, multithreading (`rayon`), HTTP server (`axum`).
*   **TFHE/LWE**: Full custom implementation of crypto logic.
*   **BigInt**: Client-side cryptography in pure JavaScript.
*   **Neural Network**: Custom `Vec<f64>` / `Vec<u64>` implementation (no Torch/TensorFlow dependencies).
