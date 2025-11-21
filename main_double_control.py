import pygame
import mediapipe as mp
import cv2
import numpy as np


# FUNCTION
def euclidean_dist(pts1,pts2):
    p1 = np.array([pts1.x,pts1.y,pts1.z])
    p2 = np.array([pts2.x, pts2.y, pts2.z])
    return np.linalg.norm(p1 - p2)


def is_fist(pts1,pts2):
    pass

def restart(pts1,pts2) -> bool:
    return bool(euclidean_dist(pts1, pts2) < 0.05)


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


x, y = 100, 100
snake = [(x,y)] * 50
STEP_MAX = 10
STEP_MIN = 2
step = STEP_MIN
direction = "Right"

# End texte
font = pygame.font.SysFont('Arial',36)
end_display = font.render("You Lose ! Try again !", True, (0,0,0))
end_position = end_display.get_rect(center=(400,100))
restart_display = font.render("Make the 'OK' sign with your hand to restart !", True, (0,0,0))
restart_position = restart_display.get_rect(center=(400,300))
speed_display = font.render(f"Speed : {step}", True, (0,0,0))
speed_position = speed_display.get_rect(center=(400,300))


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
        # tip
        thumb_tip = 4
        index_finger_tip = 8
        hand_wrist = 0
        middle_finger_tip = 12
        ring_finger_tip = 16
        pinky_finger_tip = 20
        # mcp
        index_finger_mcp = 5
        middle_finger_mcp = 9
        ring_finger_mcp = 13
        pinky_finger_mcp = 17
        
        right_hand = None
        left_hand = None

        # Detect lleft or/and right hand(s)
        for hand_idx, hand_landmarks in enumerate(points.multi_hand_landmarks):
            hand_label = points.multi_handedness[hand_idx].classification[0].label  # "Left" ou "Right"
            
            if hand_label == "Right":
                right_hand = hand_landmarks.landmark
            elif hand_label == "Left":
                left_hand = hand_landmarks.landmark

        if right_hand:
            # Direction
            thumb_1 = right_hand[thumb_tip]
            index_tip_1 = right_hand[index_finger_tip]
            wrist_1 = right_hand[hand_wrist]
            
            # calculate for directions (hand 1)
            x_wrist = int(wrist_1.x * width)
            y_wrist = int(wrist_1.y * height)
            x_index = int(index_tip_1.x * width)
            y_index = int(index_tip_1.y * height)

            if direction == None:
                # Restart (index_tip near thumb)
                if restart(thumb_1, index_tip_1):    
                    direction = "Right"
                    x, y = 100, 100
                    snake = [(x,y)] * 50
                    step = STEP_MIN
                
            else :
                # Directions (hand 1)
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


        if left_hand:
            # Speed
            thumb_2 = left_hand[thumb_tip]
            index_tip_2 = left_hand[index_finger_tip]
            middle_tip_2 = left_hand[middle_finger_tip]
            ring_tip_2 = left_hand[ring_finger_tip]
            pinky_tip_2 = left_hand[pinky_finger_tip]
            wrist_2 = left_hand[hand_wrist]

            index_mcp_2 = left_hand[index_finger_mcp]
            middle_mcp_2 = left_hand[middle_finger_mcp]
            ring_mcp_2 = left_hand[ring_finger_mcp]
            pinky_mcp_2 = left_hand[pinky_finger_mcp]


            if direction == None:
                # Restart (index_tip near thumb)
                if restart(thumb_2, index_tip_2):    
                    direction = "Right"
                    x, y = 100, 100
                    snake = [(x,y)] * 50
                    step = STEP_MIN
    
            else :
                # Speeds (hand 2)
                if euclidean_dist(index_mcp_2, index_tip_2) < 0.05 and euclidean_dist(middle_mcp_2, middle_tip_2) < 0.05 and euclidean_dist(ring_mcp_2, ring_tip_2) < 0.05 and euclidean_dist(pinky_mcp_2, pinky_tip_2) < 0.05:
                    # fist
                    if step < STEP_MAX:
                        step += 1
                if euclidean_dist(index_mcp_2, index_tip_2) > 0.1 and euclidean_dist(middle_mcp_2, middle_tip_2) > 0.1 and euclidean_dist(ring_mcp_2, ring_tip_2) > 0.1 and euclidean_dist(pinky_mcp_2, pinky_tip_2) > 0.1:
                    # open hand
                    if step > STEP_MIN:
                        step -= 1
                

           

            
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
        
        # # update snake
        # snake.pop(0)
        # snake.append((x,y))
        # Nouvelle position
        new_pos = (x, y)

        # Interpoler si step > 1
        last_pos = snake[-1]
        interpolated_positions = []
        for i in range(1, step + 1):
            interp_x = int(last_pos[0] + (new_pos[0] - last_pos[0]) * i / step)
            interp_y = int(last_pos[1] + (new_pos[1] - last_pos[1]) * i / step)
            interpolated_positions.append((interp_x, interp_y))

        # Ajouter les positions interpolées et garder la taille du serpent
        for pos in interpolated_positions:
            snake.pop(0)
            snake.append(pos)


    # UPDATE PYGAME
    screen.fill((255, 255, 255))
    speed_display = font.render(f"Speed : {step}", True, (0,0,0))
    screen.blit(speed_display,speed_position)

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


