from utils import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


env = CartPoleEnvManager("cpu")
env.reset()

screen = env.get_processed_screen()

plt.figure()
screen = screen.squeeze(0).permute(1, 2, 0)
print(screen.shape)
plt.imshow(screen, interpolation='none')
plt.show()
