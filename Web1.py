from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

cap = cv2.VideoCapture(1)
altura_objetos_cm = {"planta_1": 20.0, "planta_2": 15.0}  # Puedes agregar más plantas aquí
relaciones_px_cm = {"planta_1": None, "planta_2": None}  # Relaciones píxeles/centímetros
plantas_detectadas = {"planta_1": False, "planta_2": False}  # Estado de detección de plantas

def detectar_y_medir(frame):
    global relaciones_px_cm

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    verde_bajo = np.array([40, 40, 40])
    verde_alto = np.array([80, 255, 255])
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for nombre_planta, altura_objeto_cm in altura_objetos_cm.items():
        # Si la planta no ha sido detectada, intentamos encontrarla
        if not plantas_detectadas[nombre_planta] and contornos:
            contorno_planta = max(contornos, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contorno_planta)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            altura_objeto_px = h

            if altura_objeto_px != 0 and altura_objeto_cm != 0:
                relaciones_px_cm[nombre_planta] = altura_objeto_px / altura_objeto_cm

                altura_planta_cm = h / relaciones_px_cm[nombre_planta]

                cv2.putText(frame, f'Altura de {nombre_planta}: {altura_planta_cm:.2f} cm', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Marcamos la planta como detectada
                plantas_detectadas[nombre_planta] = True

    return frame

def gen_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame_con_medicion = detectar_y_medir(frame)
            suc, encode = cv2.imencode('.jpg', frame_con_medicion)
            frame_con_medicion = encode.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_con_medicion + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
