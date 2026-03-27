#!/usr/bin/env python3
"""
Прототип системы детекции уборки столиков по видео.
Использует YOLOv8 для обнаружения людей и ручной выбор ROI одного столика.
Отслеживает состояния "пусто" / "занято", фиксирует временные метки событий,
вычисляет среднее время между уходом гостя и подходом следующего человека.
"""

import argparse
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Detect table cleaning events')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model file')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Загрузка модели YOLO
    model = YOLO(args.model)

    # Открываем видео
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {args.video}")

    # Получаем параметры видео для записи
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Выбор области столика (ROI)
    print("Нажмите и перетащите прямоугольник вокруг столика, затем нажмите ENTER или SPACE.")
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр")
        return
    roi = cv2.selectROI("Select Table", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Table")
    x, y, w, h = roi
    print(f"Выбран столик: x={x}, y={y}, w={w}, h={h}")

    # Инициализация переменных для детекции
    state = "empty"          # empty или occupied
    last_change_time = None  # время последнего изменения состояния (в секундах)
    events = []              # список словарей с событиями
    # Для поиска пар "уход -> следующий подход"
    empty_times = []         # времена, когда столик стал пустым
    approach_times = []      # времена, когда столик стал занятым (подход)

    # Функция проверки наличия человека в ROI
    def person_in_roi(frame):
        # Выполняем детекцию на всем кадре (можно ограничить ROI для ускорения)
        results = model(frame, conf=args.conf, classes=[0])  # класс 0 = человек
        boxes = results[0].boxes
        if boxes is None:
            return False
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Проверяем пересечение bounding box с ROI
            roi_box = [x, y, x+w, y+h]
            if (x1 < roi_box[2] and x2 > roi_box[0] and
                y1 < roi_box[3] and y2 > roi_box[1]):
                return True
        return False

    # Основной цикл обработки кадров
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Текущее время в секундах от начала видео
        current_time = frame_idx / fps

        # Детекция наличия человека в ROI
        person_present = person_in_roi(frame)
        print(f"Кадр {frame_idx}: person_present = {person_present}, state = {state}")

        # Логика смены состояний
        if state == "empty" and person_present:
            # Подход к столу
            state = "occupied"
            events.append({
                "time": current_time,
                "event": "подход",
                "state_before": "empty",
                "state_after": "occupied"
            })
            approach_times.append(current_time)
            last_change_time = current_time
            print(f"{current_time:.2f}s: ПОДХОД к столу")
        elif state == "occupied" and not person_present:
            # Уход от стола
            state = "empty"
            events.append({
                "time": current_time,
                "event": "уход",
                "state_before": "occupied",
                "state_after": "empty"
            })
            empty_times.append(current_time)
            last_change_time = current_time
            print(f"{current_time:.2f}s: УХОД от стола")

        # Визуализация
        # Рисуем прямоугольник ROI с цветом в зависимости от состояния
        color = (0, 255, 0) if state == "empty" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # Добавляем текст состояния
        text = "Empty" if state == "empty" else "Occupied"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Запись кадра в выходное видео
        out.write(frame)

        # Для отображения в реальном времени (опционально)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Постобработка: сопоставление уходов и подходов
    # Для каждого ухода (стол стал пустым) ищем следующий подход
    delays = []
    i = 0
    for empty_time in empty_times:
        # Находим первый подход, который произошёл после этого ухода
        next_approach = None
        for approach_time in approach_times:
            if approach_time > empty_time:
                next_approach = approach_time
                break
        if next_approach is not None:
            delay = next_approach - empty_time
            delays.append(delay)
            print(f"Уход в {empty_time:.2f}s -> подход в {next_approach:.2f}s, задержка = {delay:.2f}s")

    # Вычисление средней задержки
    if delays:
        avg_delay = sum(delays) / len(delays)
        print(f"\nСреднее время между уходом и следующим подходом: {avg_delay:.2f} секунд")
    else:
        avg_delay = None
        print("Не найдено пар уход-подход для вычисления средней задержки.")

    # Сохранение отчёта в CSV
    df = pd.DataFrame(events)
    if not df.empty:
        df.to_csv("events.csv", index=False)
        print("События сохранены в events.csv")
    else:
        print("Событий не зафиксировано.")

    # Сохранение средней задержки в текстовый файл
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(f"Среднее время между уходом и следующим подходом: {avg_delay:.2f} секунд\n")
        f.write(f"Всего событий уходов: {len(empty_times)}\n")
        f.write(f"Всего событий подходов: {len(approach_times)}\n")
        if delays:
            f.write("Задержки (сек): " + ", ".join(f"{d:.2f}" for d in delays) + "\n")
    print("Отчёт сохранён в report.txt")

if __name__ == "__main__":
    main()