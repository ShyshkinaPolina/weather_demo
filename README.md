# Streamlit Rain Prediction App

Цей проєкт демонструє, як створити веб-застосунок на базі Streamlit для прогнозу, чи буде дощ завтра в Австралії, використовуючи дані про погоду. Користувач може ввести метеорологічні показники (температуру, вологість, напрямок вітру тощо), після чого модель зробить прогноз на основі навченої класифікаційної моделі.

Протестувати застосунок можна за посиланням (при наявності):
[https://app-aussie-rain-demo.streamlit.app/](https://app-aussie-rain-demo.streamlit.app/)

> Якщо з'являється повідомлення "This app has gone to sleep...", натисніть "Yes, get this app back up!" і зачекайте кілька секунд.

---

## 📁 Структура проєкту

* **data/**: Містить вхідний датасет погоди (`weatherAUS.csv`).
* **models/**: Зберігає навчену модель машинного навчання у форматі `.joblib`.
* **app.py**: Головний файл застосунку Streamlit.
* **requirements.txt**: Список необхідних Python-бібліотек.

---

## ⚙️ Налаштування

### ✅ Передумови

* Python 3.8 або новіший.
* Пакети з `requirements.txt`.

---

### 🧪 Встановлення

1. **Клонуйте репозиторій:**

   ```bash
   git clone https://github.com/ShyshkinaPolina/weather_demo.git
   cd weather_demo
   ```

2. **Створіть віртуальне середовище (опційно):**

   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```

3. **Встановіть залежності:**

   ```bash
   pip install -r requirements.txt
   ```

---

### 🚀 Запуск Streamlit-додатку

```bash
streamlit run app.py
```

Додаток буде доступний за адресою:
[http://localhost:8501](http://localhost:8501)

---
