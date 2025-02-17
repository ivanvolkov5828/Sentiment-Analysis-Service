# Sentiment Analysis Service

Этот проект предоставляет веб-сервис для анализа сентимента текста. Пользователь вводит текст в веб-интерфейсе, а сервис определяет, является ли текст позитивным или негативным.

## Установка и запуск

1. Клонируйте репозиторий:

```bash
git clone <repository-url>
cd sentiment_analysis_service
```

2. Создание виртуального окружения:

```bash
python -m venv venv
source venv/bin/activate  # для Windows используйте `venv\Scripts\activate`
```

3. Установите необходимые зависимости:

```bash
pip install fastapi uvicorn transformers torch
```

4. Запустите сервер с помощью команды:

```bash
uvicorn server:app --reload
```

## Использование
1. Откройте браузер и перейдите по адресу http://127.0.0.1:8000.
2. Введите текст в текстовое поле.
3. Нажмите кнопку "Analyze".
4. Результат анализа сентимента текста будет отображен на странице.

## Структура проекта

```bash
sentiment_analysis_service/
├── templates/
│   └── index.html      # HTML шаблон для веб-интерфейса
├── server.py           # Основной файл сервера
├── venv/               # Виртуальное окружение
├── README.md           # Инструкция по установке и использованию
```

## Используемые технологии
### FastAPI: Современный, быстрый (высокопроизводительный) веб-фреймворк для создания API на Python.
### Transformers: Библиотека для работы с предобученными трансформерными моделями от Hugging Face.
### Torch: Популярный фреймворк для машинного обучения.# Sentiment-Analysis-Service
