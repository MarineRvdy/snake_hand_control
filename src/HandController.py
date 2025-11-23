import mediapipe as mp
import cv2
import pygame
import numpy as np
from typing import List, Any, Dict, Tuple, Optional

# tip
THUMB_TIP: int = 4
INDEX_TIP: int = 8
MIDDLE_TIP: int = 12
RING_TIP: int = 16
PINKY_TIP: int = 20
# mcp
WRIST: int = 0
INDEX_MCP: int = 5
MIDDLE_MCP: int = 9
RING_MCP: int = 13
PINKY_MCP: int = 17


class HandController:
    
    def __init__(self) -> None:
        # Init MediaPipe
        self.__mp_hands: Any = mp.solutions.hands
        self.__hands: Any = self.__mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Init camera
        self.__cap: cv2.VideoCapture = cv2.VideoCapture(0)
        # Size camera
        self.__height: int = 0
        self.__width: int = 0

    def euclideanDistance(self, pts1: Any, pts2: Any) -> float:
        p1 = np.array([pts1.x, pts1.y, pts1.z])
        p2 = np.array([pts2.x, pts2.y, pts2.z])
        return float(np.linalg.norm(p1 - p2))

    def FrameCapture(self) -> Tuple[Optional[Any], Optional[pygame.Surface]]:
        ret: bool
        frame: Optional[np.ndarray]
        ret, frame = self.__cap.read()
        if not ret or frame is None:
            return None, None
        
        frame = cv2.flip(frame, 1)  # flip camera to mirror
        self.__height, self.__width = frame.shape[:2]
        image_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe treatment
        image_rgb.flags.writeable = False
        points: Any = self.__hands.process(image_rgb)  # hands points
        image_rgb.flags.writeable = True
        
        camera_frame: np.ndarray = np.transpose(image_rgb, (1, 0, 2))
        camera_surface: pygame.Surface = pygame.surfarray.make_surface(camera_frame)
        camera_small: pygame.Surface = pygame.transform.scale(camera_surface, (200, 150))

        return points, camera_small

    def movementDetection(self, points: Any) -> Dict[str, Any]:
        command: Dict[str, Any] = {
            "direction": None,
            "speed_change": 0,
            "restart": False
        }

        right_hand: Optional[List[Any]] = None
        left_hand: Optional[List[Any]] = None

        if points.multi_hand_landmarks and points.multi_handedness:
            # Detect left or/and right hand(s)
            for hand_idx, hand_landmarks in enumerate(points.multi_hand_landmarks):
                hand_label: str = points.multi_handedness[hand_idx].classification[0].label
                
                if hand_label == "Right":
                    right_hand = hand_landmarks.landmark
                elif hand_label == "Left":
                    left_hand = hand_landmarks.landmark
            
            # Check for restart
            if (right_hand and self.restartDetected(right_hand)) or (left_hand and self.restartDetected(left_hand)):
               command["restart"] = True

            if right_hand and len(right_hand) == 21:
                command["direction"] = self.directionDetected(right_hand)
            
            # Check for speed
            if left_hand and len(left_hand) == 21:
                command["speed_change"] = self.speedDetected(left_hand)
        
        return command
            
    def restartDetected(self, hand: List[Any]) -> bool:
        thumb_tip: Any = hand[THUMB_TIP]
        index_tip: Any = hand[INDEX_TIP]
        return bool(self.euclideanDistance(thumb_tip, index_tip) < 0.05)

    def directionDetected(self, hand: List[Any]) -> Optional[str]:
        direction: Optional[str] = None
        index_tip: Any = hand[INDEX_TIP]
        wrist: Any = hand[WRIST]
        
        x_wrist: int = int(wrist.x * self.__width)
        y_wrist: int = int(wrist.y * self.__height)
        x_index: int = int(index_tip.x * self.__width)
        y_index: int = int(index_tip.y * self.__height)

        if abs(x_index - x_wrist) > abs(y_index - y_wrist):

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

        return direction

    def speedDetected(self, hand: List[Any]) -> int:
        change: int = 0
        fingers_tips: List[int] = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        fingers_mcps: List[int] = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

        dists: List[float] = []
        for tip, mcp in zip(fingers_tips, fingers_mcps):
            dists.append(self.euclideanDistance(hand[tip], hand[mcp]))
        
        # Fist
        if all(d < 0.07 for d in dists): 
            change = 1
        # Open hand
        if all(d > 0.14 for d in dists):
            change = -1

        return change

    def close(self) -> None:
        self.__cap.release()