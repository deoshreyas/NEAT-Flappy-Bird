import neat.statistics
import pygame
from pygame.locals import *
import neat
import time
import os
import random

pygame.init()
pygame.font.init()

# Initializing window
WIN_WIDTH = 500
WIN_HEIGHT = 600

GEN = 0

win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird AI")

# Load images
BIRD_IMGS = [
    pygame.image.load(os.path.join("assets", "player0.png")), 
    pygame.image.load(os.path.join("assets", "player1.png")), 
    pygame.image.load(os.path.join("assets", "player2.png"))
]
PIPE_IMG = pygame.image.load(os.path.join("assets", "pipe.png"))
GROUND_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("assets", "base.png")))
BG_IMG = pygame.transform.scale(pygame.image.load(os.path.join("assets", "background.png")), (WIN_WIDTH, WIN_HEIGHT))

# Fonts 
STAT_FONT = pygame.font.Font("assets/Silkscreen-Regular.ttf", 25)

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]
    
    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y
    
    def move(self):
        self.tick_count += 1

        # Calculate displacement
        d = self.vel*self.tick_count + 1.5*self.tick_count**2

        # Terminal velocity
        if d >= 16:
            d = 16
        
        # Jump height
        if d < 0:
            d -= 2
        
        self.y = self.y + d

        # Tilt the bird
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        # Flap wings
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0
        
        # Don't flap wings when going down
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2
        
        # Rotate the bird
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 135
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()
    
    def set_height(self):
        self.height = random.randrange(50, 300)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
    
    def move(self):
        self.x -= self.VEL
    
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))
    
    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True
        
        return False

class Ground:
    VEL = 5
    WIDTH = GROUND_IMG.get_width()
    IMG = GROUND_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

def draw_window(win, birds, pipes, base, score, gen, alive):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    txt = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(txt, (WIN_WIDTH - 10 - txt.get_width(), 10))
    txt = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(txt, (10, 10))
    txt = STAT_FONT.render("Alive: " + str(alive), 1, (255, 255, 255))
    win.blit(txt, (10, 40))

    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()

def main(genomes, config):
    global GEN 
    GEN += 1

    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(75, 270))
        g.fitness = 0
        ge.append(g)

    ground = Ground(482)
    pipes = [Pipe(WIN_WIDTH + 100)]
    clock = pygame.time.Clock()

    score = 0

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                pygame.quit()
                quit()
        
        pipe_ind = 0
        if len(birds)>0:
            if len(pipes)>1 and birds[0].x>pipes[0].x+pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            running = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
            
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()
        
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))
        
        for r in rem:
            pipes.remove(r)

        for x,bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 482 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        ground.move()
        draw_window(win, birds, pipes, ground, score, GEN, len(birds))


        pygame.display.update()

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)