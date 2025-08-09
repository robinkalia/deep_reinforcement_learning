import math
from torch.utils.tensorboard.writer import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter()
    functions = {"sin": math.sin, "cos": math.cos, "tan": math.tan}

    for angle in range(-720,720):
        angle_rad = angle * math.pi / 180.0
        for name, func in functions.items():
            val = func(angle_rad)
            writer.add_scalar(name, val, angle)

    print("Finished Tensorboard test operation")
    writer.close()