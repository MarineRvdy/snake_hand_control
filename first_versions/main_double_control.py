import pygame
import mediapipe as mp
import cv2
import numpy as np


# CSTE
# tip
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
# mcp
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

# FUNCTION
def euclidean_dist(pts1,pts2):
    p1 = np.array([pts1.x,pts1.y,pts1.z])
    p2 = np.array([pts2.x, pts2.y, pts2.z])
    return np.linalg.norm(p1 - p2)


def is_fist(pts1,pts2):
    pass

def restart(hand) -> bool:
    thumb_tip = hand[THUMB_TIP]
    index_tip = hand[INDEX_TIP]
    return bool(euclidean_dist(thumb_tip, index_tip) < 0.05)


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
cap = cv2.VideoCapture(0)  # 0 = webcam

# PYGAME INIT
pygame.init()

# Windows creation
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Snake hand control")

# Variable de contrôle
running = True
clock = pygame.time.Clock()

# INIT VARIABLE
x, y = 500, 100
snake = [(x,y)] * 50
STEP_MAX = 10
STEP_MIN = 2
step = STEP_MIN
direction = "Left"
game_over = False

# TEXT DISPLAY
font = pygame.font.SysFont('Arial',36)
end_display = font.render("You Lose ! Try again !", True, (0,0,0))
end_position = end_display.get_rect(center=(400,100))
restart_display = font.render("Make the 'OK' sign with your hand to restart !", True, (0,0,0))
restart_position = restart_display.get_rect(center=(400,300))
speed_display = font.render(f"Speed : {step}", True, (0,0,0))
speed_position = speed_display.get_rect(center=(400,300))


# WHILE CAMERA and PYGAME
while running:   
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) # flip camera to mirror
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image.flags.writeable = False
    points = hands.process(image) # hands points
    image.flags.writeable = True
    
    # camera_frame = np.swapaxes(image, 0, 1)
    camera_frame = np.transpose(image, (1, 0, 2))   # SANS rotation
    camera_surface = pygame.surfarray.make_surface(camera_frame)
    camera_small = pygame.transform.scale(camera_surface, (200, 150))
    
    height, width = image.shape[:2]


    # Draw hand landmarks
    if points.multi_hand_landmarks and points.multi_handedness:
        
        right_hand = None
        left_hand = None
        
        # Detect left or/and right hand(s)
        for hand_idx, hand_landmarks in enumerate(points.multi_hand_landmarks):
            hand_label = points.multi_handedness[hand_idx].classification[0].label  # "Left" ou "Right"
            
            if hand_label == "Right":
                right_hand = hand_landmarks.landmark
            elif hand_label == "Left":
                left_hand = hand_landmarks.landmark

        # Check for restart if game over
        if game_over:
            if (right_hand and restart(right_hand)) or (left_hand and restart(left_hand)):
                direction = "Left"
                x, y = 500, 100
                snake = [(x,y)] * 50
                step = STEP_MIN
                game_over = False

        # Right hand Direction
        if right_hand and len(right_hand) == 21:
            
            index_tip = right_hand[INDEX_TIP]
            wrist = right_hand[WRIST]
            
            # calculate
            x_wrist = int(wrist.x * width)
            y_wrist = int(wrist.y * height)
            x_index = int(index_tip.x * width)
            y_index = int(index_tip.y * height)


            if abs(x_index-x_wrist) > abs(y_index-y_wrist):

                if x_index > x_wrist:
                    # Right
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

        # Left hand Speed
        if left_hand and len(left_hand) == 21:

            thumb_tip = left_hand[THUMB_TIP]
            index_tip = left_hand[INDEX_TIP]
            middle_tip = left_hand[MIDDLE_TIP]
            ring_tip = left_hand[RING_TIP]
            pinky_tip = left_hand[PINKY_TIP]
            wrist = left_hand[WRIST]

            index_mcp = left_hand[INDEX_MCP]
            middle_mcp = left_hand[MIDDLE_MCP]
            ring_mcp = left_hand[RING_MCP]
            pinky_mcp = left_hand[PINKY_MCP]

          
            if euclidean_dist(index_mcp, index_tip) < 0.07 and euclidean_dist(middle_mcp, middle_tip) < 0.07 and euclidean_dist(ring_mcp, ring_tip) < 0.07 and euclidean_dist(pinky_mcp, pinky_tip) < 0.07:
                # fist
                if step < STEP_MAX:
                    step += 1
            if euclidean_dist(index_mcp, index_tip) > 0.14 and euclidean_dist(middle_mcp, middle_tip) > 0.14 and euclidean_dist(ring_mcp, ring_tip) > 0.14 and euclidean_dist(pinky_mcp, pinky_tip) > 0.14:
                # open hand
                if step > STEP_MIN:
                    step -= 1
                

           

            
    # Snake movement
    if not(game_over):
        match direction:
            case "Right":
                x += step
            case "Left":
                x -= step
            case "Up":
                y -= step
            case "Down":
                y += step

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
    screen.blit(camera_small,(0,450))
    screen.blit(speed_display,speed_position)
    

    if x <= 0 or x >= 799 or y <= 0 or y >= 599:
        screen.fill((255,0,0))
        screen.blit(camera_small,(0,450))
        screen.blit(end_display,end_position)
        screen.blit(restart_display,restart_position)
        game_over = True
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

    # QUIT PYGAME
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

cap.release()
cv2.destroyAllWindows()
pygame.quit()


