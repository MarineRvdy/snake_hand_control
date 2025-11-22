import cv2
import pygame
from HandController import HandController
from Snake import Snake

class Game:
    def __init__(self):
        pygame.init()
        self.__height = 600
        self.__width = 800
        self.__screen = pygame.display.set_mode((self.__width,self.__height))
        pygame.display.set_caption("Snake hand control")
        self.__font = pygame.font.SysFont('Arial', 36)
               


        # Control variable
        self.__clock = pygame.time.Clock()
        self.__running : bool = True
        self.__game_over : bool = False
        self.__camera_image = None

        # Init HandController and Snake
        self.__hand_controller = HandController()
        self.__snake : Snake = Snake(500, 100)
    
    def handleInput(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.__running = False

    def update(self):
        
        points, self.__camera_image = self.__hand_controller.FrameCapture()
        
        if points is None: 
            return
        

        commands = self.__hand_controller.movementDetection(points)

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

    def gameDisplay(self):
        self.__screen.fill((255, 255, 255))

        # display camera
        if self.__camera_image != None:
            self.__screen.blit(self.__camera_image, (0, 450))
        
        # display game over screen
        if self.__game_over:
            self.gameOverDisplay()
        
        # display running screen
        else:
            self.__snake.draw(self.__screen)
            self.gameRunningDisplay()

        pygame.display.flip()

    def gameOverDisplay(self):
        self.__screen.fill((255, 0, 0))
        
        if self.__camera_image != None:
            self.__screen.blit(self.__camera_image, (0, 450))

        end_display = self.__font.render("You Lose ! Try again !", True, (0,0,0))
        restart_display = self.__font.render("Make the 'OK' sign with your hand to restart !", True, (0,0,0))
        self.__screen.blit(end_display, end_display.get_rect(center=(400, 100)))
        self.__screen.blit(restart_display, restart_display.get_rect(center=(400, 300)))

    def gameRunningDisplay(self):
        speed_display = self.__font.render(f"Speed : {self.__snake._Snake__speed}", True, (0,0,0))
        self.__screen.blit(speed_display, (350, 550))
    
    def run(self):
        while self.__running:
            self.handleInput()
            self.update()
            self.gameDisplay()
            self.__clock.tick(60)

        # quit
        self.__hand_controller.close()
        cv2.destroyAllWindows()
        pygame.quit()
