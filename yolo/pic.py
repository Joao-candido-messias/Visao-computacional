import time
import cv2

COLORS = [(0, 255, 255), (255,255,0), (0,255,0), (0,0,255)] 

class_names = []

with open("coco.names" , "r") as file:
    class_names = [cname.strip() for cname in file]

imagem = "dog.jpg"

frame = cv2.imread(imagem)

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (416, 416), scale = 1/255)

start = time.time()

classes, scores, boxes = model.detect(frame, 0.1, 0.2)

end = time.time()

# Loop sobre todas as detecções
for classid, score, box in zip(classes, scores, boxes):
    # Gerando uma cor para a classe
    color = COLORS[int(classid) % len(COLORS)]

    # Pegando o nome da classe pelo id e seu score de acurácia
    label = f"{class_names[int(classid)]} : {score:.2f}"

    # Desenhando a caixa da detecção
    cv2.rectangle(frame, box, color, 2)

    # Escrevendo o nome da classe acima da caixa
    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


fps_label = f"FPS: {round((1.0/(end- start)), 2)}"

cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

cv2.imshow("detections", frame)

cv2.waitKey(0)

cv2.destroyAllWindows()