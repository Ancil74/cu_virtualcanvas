import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

# Virtual Drawing Application
# Controls: Use the index finger to draw or select toolbar buttons.
# Keyboard shortcuts: C = clear canvas, S = save, Q = quit.

CAMERA_INDEX = 0
BRUSH_SIZE = 8
ERASER_SIZE = 45

COLORS = {
    "White": (255, 255, 255),
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
    "Purple": (255, 0, 255),
}

def fingers_up(hand_landmarks):
    """Return whether the index finger is extended (simple fingertip test)."""
    lm = hand_landmarks.landmark
    # In image coordinates, a smaller y value is higher on the screen.
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y
    return index_up, middle_up, ring_up, pinky_up

def save_drawing(canvas):
    filename = "drawing_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    # Save the drawing layer with a white background so it opens as a normal image.
    image = np.full_like(canvas, 255)
    mask = np.any(canvas != 0, axis=2)
    image[mask] = canvas[mask]
    cv2.imwrite(filename, image)
    print(f"Drawing saved as: {filename}")

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check that a camera is connected.")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils

    ok, frame = cap.read()
    if not ok:
        print("Error: Could not read from webcam.")
        cap.release()
        return

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    previous_point = None
    selected_color = COLORS["Blue"]
    eraser_mode = False
    last_button_action = 0.0
    button_cooldown = 0.7

    # Toolbar buttons: (label, x1, y1, x2, y2)
    buttons = []
    x = 10
    for name in COLORS:
        buttons.append((name, x, 10, x + 82, 55))
        x += 88
    buttons += [
        ("ERASER", x + 2, 10, x + 94, 55),
        ("CLEAR", x + 100, 10, x + 190, 55),
        ("SAVE", x + 196, 10, x + 276, 55),
    ]

    print("Virtual Drawing Application started.")
    print("Raise only your index finger to draw. Touch toolbar buttons to use them.")
    print("Keyboard: C = clear, S = save, Q = quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        index_point = None
        draw_gesture = False

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark[8]
            index_point = (int(lm.x * width), int(lm.y * height))
            index_up, middle_up, ring_up, pinky_up = fingers_up(hand)
            # Draw with index finger only. Other fingers down prevents accidental strokes.
            draw_gesture = index_up and not middle_up and not ring_up and not pinky_up
            cv2.circle(frame, index_point, 10, (0, 255, 255), cv2.FILLED)

        # Draw toolbar on the camera preview.
        for label, x1, y1, x2, y2 in buttons:
            if label in COLORS:
                color = COLORS[label]
                active = (selected_color == color and not eraser_mode)
                fill = color if active else (55, 55, 55)
            elif label == "ERASER":
                fill = (120, 120, 120) if eraser_mode else (55, 55, 55)
            elif label == "CLEAR":
                fill = (0, 0, 180)
            else:
                fill = (0, 130, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, cv2.FILLED)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (230, 230, 230), 1)
            text_color = (0, 0, 0) if label in COLORS and fill == COLORS[label] else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 5, y1 + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

        # Touch a toolbar button with the index fingertip to activate it.
        button_touched = False
        if index_point is not None:
            px, py = index_point
            for label, x1, y1, x2, y2 in buttons:
                if x1 <= px <= x2 and y1 <= py <= y2:
                    button_touched = True
                    now = time.time()
                    if now - last_button_action >= button_cooldown:
                        if label in COLORS:
                            selected_color = COLORS[label]
                            eraser_mode = False
                        elif label == "ERASER":
                            eraser_mode = True
                        elif label == "CLEAR":
                            canvas[:] = 0
                            previous_point = None
                        elif label == "SAVE":
                            save_drawing(canvas)
                        last_button_action = now
                    break

        # Draw only below the toolbar and when the index-only gesture is active.
        if index_point is not None and draw_gesture and not button_touched and index_point[1] > 65:
            if previous_point is not None:
                if eraser_mode:
                    cv2.line(canvas, previous_point, index_point, (0, 0, 0),
                             ERASER_SIZE, cv2.LINE_AA)
                else:
                    cv2.line(canvas, previous_point, index_point, selected_color,
                             BRUSH_SIZE, cv2.LINE_AA)
            previous_point = index_point
        else:
            previous_point = None

        # Combine camera feed and drawing layer.
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inverse_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        drawing_layer = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(inverse_mask))
        display = cv2.addWeighted(frame, 1, drawing_layer, 1, 0)

        mode = "ERASER" if eraser_mode else "DRAW"
        cv2.putText(display, f"Mode: {mode} | C: Clear  S: Save  Q: Quit",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Virtual Drawing Application", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
            previous_point = None
        elif key == ord("s"):
            save_drawing(canvas)

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
