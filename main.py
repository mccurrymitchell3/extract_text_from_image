import cv2
import pytesseract

pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = cv2.imread("paragraph.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 1)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 5))
dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)


contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

im2 = img.copy()

file = open("recognized.txt", "w+")
file.write("")
file.close()

cv2.drawContours(dilation, contours, -1, (0, 255, 0), 3) 
  
cv2.imshow('Contours', dilation) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt) 
      
    # Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w] 
      
    # Open the file in append mode 
    file = open("recognized.txt", "a") 
      
    # Apply OCR on the cropped image 
    text = pytesseract.image_to_string(cropped) 
      
    # Appending the text into file 
    file.write(text) 
    file.write("\n") 
      
    # Close the file 
    file.close 