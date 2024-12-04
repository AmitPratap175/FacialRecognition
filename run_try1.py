import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Initialize MTCNN for face detection and ResNet for face recognition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)        #'vggface2'

# Define transformation to convert PIL image to tensor
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# Load a saved image and calculate its embedding
saved_image_path = "amit_1.jpeg"  # Path to your saved image
img = Image.open(saved_image_path)
img.save('amit_1.jpg', 'JPEG')
saved_image_path = "amit_1.jpg"
saved_image = Image.open(saved_image_path)

# Resize the image and apply transformation
saved_image = transform(saved_image).unsqueeze(0).to(device)  # Add batch dimension and move to device
# print('Shape:', saved_image.shape)
saved_embedding = resnet(saved_image)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    try:
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                
                # Apply the same transformation to the face
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                # Detect faces using MTCNN
                face_cropped = mtcnn(face_pil)

                # Ensure face is detected properly before proceeding
                if face_cropped is not None:
                    face_cropped = face_cropped.squeeze(0)  # Adjust input dimensions
                    face_embedding = resnet(face_cropped.unsqueeze(0).to(device))

                    # Compute cosine similarity between embeddings
                    similarity = torch.nn.functional.cosine_similarity(saved_embedding, face_embedding).item()

                    # print(f'Similarity: {similarity}')

                    # Display green for a match (similarity > threshold), red otherwise
                    color = (0, 255, 0) if similarity > 0.5 else (0, 0, 255)  # Threshold set at 0.6

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow('Face Recognition', frame)

    except Exception as e:
        print("Error:", e)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
