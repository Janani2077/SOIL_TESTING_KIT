# Portable Soil Testing Kit

## Overview

Portable Soil Testing Kit is an IoT and Machine Learning-based application developed to analyze soil health using sensor data. The system collects soil parameters through an Arduino, transfers the data to a Django web application via USB serial communication, and generates insights such as soil quality, soil type, crop recommendations, and fertilizer suggestions.

## Features

- Soil pH analysis
- Soil moisture measurement
- NPK nutrient analysis
- Electrical Conductivity (EC) monitoring
- Soil type prediction
- Soil quality classification
- Soil Health Score generation
- Crop recommendation
- Fertilizer recommendation
- Interactive dashboard
- Historical report storage

## Tech Stack

Hardware
- Arduino
- Soil Moisture Sensor
- pH Sensor
- NPK Sensor
- EC Sensor

Software
- Python
- Django
- SQLite
- HTML
- CSS
- JavaScript
- Chart.js

Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Random Forest Classifier
- Random Forest Regressor

## Installation

Clone the repository

```bash

```

Navigate to the project

```bash
cd Portable-Soil-Testing-Kit
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Apply migrations

```bash
python manage.py migrate
```

Run the project

```bash
python manage.py runserver
```

Open the application at

```
http://127.0.0.1:8000/
```

## Project Structure

```text
Portable-Soil-Testing-Kit/
│
├── arduino/
├── ml_models/
├── django_app/
├── static/
├── templates/
├── db.sqlite3
├── requirements.txt
└── README.md
```

## Future Improvements

- Wireless communication using ESP32
- Mobile application support
- Cloud database integration
- Weather API integration
- GPS-based soil mapping