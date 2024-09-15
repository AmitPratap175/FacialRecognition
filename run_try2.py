import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os

# Initialize MTCNN for face detection and ResNet for face recognition
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define transformation to convert PIL image to tensor
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# Load and process reference images
reference_images_path = 'reference_img'
reference_embeddings = []
reference_labels = []

for filename in os.listdir(reference_images_path):
    if filename.lower().endswith(('.jpeg', '.jpg')):
        file_path = os.path.join(reference_images_path, filename)
        
        # Convert .jpeg to .jpg if necessary
        if filename.lower().endswith('.jpeg'):
            img = Image.open(file_path)
            jpg_path = file_path.rsplit('.', 1)[0] + '.jpg'
            img.save(jpg_path, 'JPEG')
            file_path = jpg_path
        
        # Load and process the image
        img = Image.open(file_path)
        img = transform(img).unsqueeze(0).to(device)
        embedding = resnet(img)
        
        # Store embedding and label
        reference_embeddings.append(embedding)
        reference_labels.append(filename)

# Convert reference embeddings to tensor
reference_embeddings = torch.cat(reference_embeddings).detach()

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
                    face_embedding = resnet(face_cropped.unsqueeze(0).to(device)).detach()

                    # Compute cosine similarity between embeddings
                    similarities = torch.nn.functional.cosine_similarity(reference_embeddings, face_embedding).cpu().numpy()
                    best_match_index = np.argmax(similarities)
                    best_match_similarity = similarities[best_match_index]
                    best_match_label = reference_labels[best_match_index]

                    # Display green for a match (similarity > threshold), red otherwise
                    color = (0, 255, 0) if best_match_similarity > 0.5 else (0, 0, 255)  # Threshold set at 0.5

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Display label text
                    if best_match_similarity > 0.5:
                        text = best_match_label.split('.')[0]
                        # Define font and position
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        font_color = (255, 255, 255)  # White text
                        font_thickness = 2
                        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                        text_x = x1
                        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
                        # Put text on the frame
                        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)

    except Exception as e:
        print("Error:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
