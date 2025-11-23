import cv2
import pygame
from HandController import HandController
from Snake import Snake
from typing import Any, Dict, Optional

class Game:
    def __init__(self) -> None:
        pygame.init()
        self.__height: int = 600
        self.__width: int = 800
        self.__screen: pygame.Surface = pygame.display.set_mode((self.__width, self.__height))
        pygame.display.set_caption("Snake hand control")
        self.__font: pygame.font.Font = pygame.font.SysFont('Arial', 36)

        self.__clock: pygame.time.Clock = pygame.time.Clock()
        self.__running: bool = True
        self.__game_over: bool = False
        self.__camera_image: Optional[pygame.Surface] = None

        self.__hand_controller: HandController = HandController()
        self.__snake: Snake = Snake(500, 100)
    
    def handleInput(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__running = False

    def update(self) -> None:
        
        points, self.__camera_image = self.__hand_controller.FrameCapture()
        
        if points is None:
            return
        
        commands: Dict[str, Any] = self.__hand_controller.movementDetection(points)

        if self.__game_over:
            # restart management
            if commands["restart"]:
                self.__snake.reset()
                self.__game_over = False
        else:
            # commands management
            if commands["speed_change"] != 0:
                self.__snake.speedChange(commands["speed_change"])
            
            self.__snake.update(commands["direction"])

            # check for collision
            if self.__snake.checkCollision(self.__width, self.__height):
                self.__game_over = True

    def gameDisplay(self) -> None:
        self.__screen.fill((255, 255, 255))

        if self.__camera_image is not None:
            self.__screen.blit(self.__camera_image, (0, 450))
        
        # display game over screen
        if self.__game_over:
            self.gameOverDisplay()
        
        # display running screen
        else:
            self.__snake.draw(self.__screen)
            self.gameRunningDisplay()

        pygame.display.flip()

    def gameOverDisplay(self) -> None:
        self.__screen.fill((255, 0, 0))
        
        if self.__camera_image is not None:
            self.__screen.blit(self.__camera_image, (0, 450))

        end_display: pygame.Surface = self.__font.render("You Lose ! Try again !", True, (0, 0, 0))
        restart_display: pygame.Surface = self.__font.render("Make the 'OK' sign with your hand to restart !", True, (0, 0, 0))
        self.__screen.blit(end_display, end_display.get_rect(center=(400, 100)))
        self.__screen.blit(restart_display, restart_display.get_rect(center=(400, 300)))

    def gameRunningDisplay(self) -> None:
        speed_display: pygame.Surface = self.__font.render(f"Speed : {self.__snake.getSpeed()}", True, (0, 0, 0))
        self.__screen.blit(speed_display, (350, 550))
    
    def run(self) -> None:
        while self.__running:
            self.handleInput()
            self.update()
            self.gameDisplay()
            self.__clock.tick(60)

        # quit
        self.__hand_controller.close()
        cv2.destroyAllWindows()
        pygame.quit()
