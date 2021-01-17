import os
import argparse
from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='./tmp/model.ckpt', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=48, type=int)
parser.add_argument('--num_slots', default=11, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=32, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_epochs', default=50, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_epochs', default=500, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=600, type=int, help='number of workers for loading data')

opt = parser.parse_args()
resolution = (128, 128)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_set = CLEVR('train')
model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)

criterion = nn.MSELoss()

params = [{'params': model.parameters()}]

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

start = time.time()
for epoch in range(opt.num_epochs):
    model.train()
    # if epoch < opt.warmup_epochs:
    #     learning_rate = opt.learning_rate * (epoch / opt.warmup_epochs)
    # else:
    learning_rate = opt.learning_rate

    learning_rate = learning_rate * (opt.decay_rate ** (
        epoch / opt.decay_epochs))
    
    optimizer = optim.Adam(params, lr=learning_rate)

    total_loss = 0

    for sample in tqdm(train_dataloader):
        image = sample['image'].to(device)
        recon_combined, recons, masks, slots = model(image)
        loss = criterion(recon_combined, image)
        total_loss += loss.item()

        del recons, masks, slots

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss /= len(train_dataloader)

    print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
        datetime.timedelta(seconds=time.time() - start)))

    # if not epoch % 10:
    torch.save({
        'model_state_dict': model.state_dict(),
        }, opt.model_dir)
