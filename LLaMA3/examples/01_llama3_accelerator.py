from accelerate import Accelerator

accelerator = Accelerator()

device = accelerator.device
model.to(device)