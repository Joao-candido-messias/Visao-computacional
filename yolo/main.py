#TESTE COM DETECÇÃO DE VIDEO

import time
import cv2

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
# Carrega as classes do coco.names e coloca no array class_names
with open("coco.names", "r") as file:
    class_names = [cname.strip() for cname in file]

# Abre a webcam ou o vídeo
cap = cv2.VideoCapture("walking.mp4")

# Carregando os dados da rede neural
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Setando os parâmetros para a rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# Lendo os frames do vídeo
while True:
    # Captura do frame
    ret, frame = cap.read()
    if not ret:
        break

    # Começo da contagem do tempo
    start = time.time()

    # Detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # Fim da contagem do tempo
    end = time.time()

    for classid, score, box in zip(classes, scores, boxes):
        # Gerando uma cor para a classe
        color = COLORS[int(classid) % len(COLORS)]

        # Pegando o nome da classe pelo id e seu score de acurácia
        label = f"{class_names[int(classid)]} : {score:.2f}"

        # Desenhando a caixa da detecção
        cv2.rectangle(frame, box, color, 4)

        # Escrevendo o nome da classe acima da caixa
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculando o FPS
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

    # Escrevendo FPS na imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Mostrando a imagem
    cv2.imshow("detections", frame)

    # Verificação de tecla para saída
    if cv2.waitKey(1) == 27:
        break

# Liberação da câmera e destruição das janelas
cap.release()
cv2.destroyAllWindows()
