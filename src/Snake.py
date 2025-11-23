import pygame
from typing import List, Any, Dict, Tuple, Optional

SPEED_MIN: int = 2
SPEED_MAX: int = 10

class Snake:
    
    def __init__(self, x_start: int, y_start: int) -> None:
        self.__x_start: int = x_start
        self.__y_start: int = y_start
        self.reset()
        
    def reset(self) -> None:
        self.__x: int = self.__x_start
        self.__y: int = self.__y_start
        
        self.__body: List[Tuple[int, int]] = [(self.__x, self.__y)] * 50
        self.__speed: int = SPEED_MIN
        self.__direction: str = "Left"

    def speedChange(self, change: int) -> None:
        if change == 1 and self.__speed < SPEED_MAX:
            self.__speed += 1
        elif change == -1 and self.__speed > SPEED_MIN:
            self.__speed -= 1

    def update(self, direction: Optional[str]) -> None:
        if direction:
            self.__direction = direction

        match self.__direction:
            case "Right":
                self.__x += self.__speed
            case "Left":
                self.__x -= self.__speed
            case "Up":
                self.__y -= self.__speed
            case "Down":
                self.__y += self.__speed

        new_pos: Tuple[int, int] = (self.__x, self.__y)
        last_pos: Tuple[int, int] = self.__body[-1]

        interpolated_positions: List[Tuple[int, int]] = []
        for i in range(1, self.__speed + 1):
            interp_x: int = int(last_pos[0] + (new_pos[0] - last_pos[0]) * i / self.__speed)
            interp_y: int = int(last_pos[1] + (new_pos[1] - last_pos[1]) * i / self.__speed)
            interpolated_positions.append((interp_x, interp_y))

        # Add positions interpolate and keep snake size
        for pos in interpolated_positions:
            self.__body.pop(0)
            self.__body.append(pos)
    
    def checkCollision(self, screen_width: int, screen_height: int) -> bool:
        collision: bool = False
        if self.__x <= 0 or self.__x >= screen_width - 1 or self.__y <= 0 or self.__y >= screen_height - 1:
            collision = True
        return collision

    def draw(self, surface: pygame.Surface) -> None:
        for position in self.__body:
            if position is not None:
                # Body
                pygame.draw.circle(surface, (0, 255, 0), position, 5)
        
        # Eyes
        pygame.draw.circle(surface, (0, 0, 0), (int(self.__x) - 3, int(self.__y) - 3), 2)
        pygame.draw.circle(surface, (0, 0, 0), (int(self.__x) + 3, int(self.__y) - 3), 2)