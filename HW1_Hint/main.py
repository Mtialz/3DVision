import cv2
url="https://192.168.110.154:8080/video"
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Error: Could not open camera stream.")
    exit()

# Loop to read and display frames
while True:
    # Read a frame from the stream
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Display the frame in a window
    cv2.imshow("IP Camera Stream", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the window
cap.release()
cv2.destroyAllWindows()