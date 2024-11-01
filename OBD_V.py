import torch 
import cv2
from ultralytics import YOLO
 
model_path = "best.pt"
confidence_threshold = 0.20  
model = YOLO(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
  
cap = cv2.VideoCapture(0)
 
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # Perform prediction on the image
    results = model(frame)
 
    # Read the image
    image = results[0].orig_img
 
    # Resize the image
    image = cv2.resize(image, (800, 600))
 
    # Get original image dimensions
    orig_height, orig_width, _ = results[0].orig_img.shape
    labes = [results[0].names[int(box)]  for box in results[0].boxes.cls]
 
    for x in range(len(results[0].boxes.xyxy)):
        xmin, ymin, xmax, ymax = map(int, results[0].boxes.xyxy[x].cpu().numpy())
 
        # Normalize coordinates
        xmin_norm = int(xmin * 800 / orig_width)
        ymin_norm = int(ymin * 600 / orig_height)
        xmax_norm = int(xmax * 800 / orig_width)
        ymax_norm = int(ymax * 600 / orig_height)
 
        # Draw bounding box
        cv2.rectangle(image, (xmin_norm, ymin_norm), (xmax_norm, ymax_norm), (0, 255, 0), 2)
        class_label = labes[x]
       
        # Add text label
        cv2.putText(image, class_label, (xmin_norm, ymin_norm - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Display the image with bounding box
    cv2.imshow("Bounding Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()