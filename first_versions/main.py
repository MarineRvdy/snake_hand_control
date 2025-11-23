import pygame
import mediapipe as mp
import cv2
import numpy as np


# FUNCTION
def euclidean_dist(pts1,pts2):
    p1 = np.array([pts1.x,pts1.y,pts1.z])
    p2 = np.array([pts2.x, pts2.y, pts2.z])
    return np.linalg.norm(p1 - p2)

# HANDS INIT
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# CAMERA INIT
cv2.namedWindow("Camera")
cap = cv2.VideoCapture(0)  # 0 = webcam

if cap.isOpened():
    ret, frame = cap.read()
else :
    ret = False


# PYGAME INIT
pygame.init()

# Windows creation
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Snake hand control")

# Variable de contrôle
running = True
clock = pygame.time.Clock()

# End texte
font = pygame.font.SysFont('Arial',36)
end_display = font.render("You Lose ! Try again !", True, (0,0,0))
end_position = end_display.get_rect(center=(400,100))
restart_display = font.render("Make the 'OK' sign with your hand to restart !", True, (0,0,0))
restart_position = restart_display.get_rect(center=(400,300))


x, y = 100, 100
snake = [(x,y)] + [None for _ in range(20)]
step = 3
direction = "Right"



# WHILE CAMERA and PYGAME
while ret and running:   
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # flip camera to mirror
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    points = hands.process(image) # hands points
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    height, width = image.shape[:2]


    # Draw hand landmarks
    if points.multi_hand_landmarks:
        # index tip
        thumb_tip = 4
        index_finger_tip = 8
        hand_wrist = 0
        

        for hand_landmarks in points.multi_hand_landmarks: # hand_landmarks contains 21 coords (x,y,z)
            landmarks = hand_landmarks.landmark
            
            # recover (x,y,z)
            thumb = landmarks[thumb_tip]
            index_tip = landmarks[index_finger_tip]
            wrist = landmarks[hand_wrist]
            
            x_wrist = int(wrist.x * width)
            y_wrist = int(wrist.y * height)
            x_index = int(index_tip.x * width)
            y_index = int(index_tip.y * height)

            if direction == None:
                # Restart (index_tip near thumb)
                if euclidean_dist(thumb, index_tip) < 0.05:    
                    direction = "Right"
                    x, y = 100, 100
                    snake = [(x,y)] + [None for _ in range(20)]
                    
            else :
                # Directions
                if abs(x_index-x_wrist) > abs(y_index-y_wrist):

                    if x_index > x_wrist:
                        direction = "Right"
                    else:
                        # Left
                        direction = "Left"
                else:
                    if y_index > y_wrist:
                        # Down
                        direction = "Down"
                    else:
                        # Up
                        direction = "Up"

           

            
    # Snake movement
    if direction != None:
        match direction:
            case "Right":
                x += step
            case "Left":
                x -= step
            case "Up":
                y -= step
            case "Down":
                y += step
        
        # update snake
        snake.pop(0)
        snake.append((x,y))

    # UPDATE PYGAME
    screen.fill((255, 255, 255))
    if x <= 0 or x >= 799 or y <= 0 or y >= 599:
        screen.fill((255,0,0))
        screen.blit(end_display,end_position)
        screen.blit(restart_display,restart_position)
        direction = None
    else :
        for position in snake:
            if position != None:
                # snake
                pygame.draw.circle(screen, (0, 255, 0), position, 5)
                # eyes
                pygame.draw.circle(screen, (0, 0, 0), (x - 3, y - 3), 2)
                pygame.draw.circle(screen, (0, 0, 0), (x + 3, y - 3), 2)
    
    pygame.display.flip()
    clock.tick(60)

    # UPDATE CAMERA
    cv2.imshow('Camera', image)
    
    # QUIT CAMERA
    key = cv2.waitKey(20)
    if key == 27:
        break
    # QUIT PYGAME
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()


