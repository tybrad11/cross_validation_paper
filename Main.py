
import torchvision
import torch
import numpy as np
from models import ConvModel, NNModel, select_model
from train_utils import test_model, train_model, kfoldcv
from torchvision import datasets, transforms
import csv
from torch.utils.data import Dataset, DataLoader
import random
import os

# parent_dir = '/home/tjb129/PycharmProjects/cross_validation_paper'
parent_dir = '/Data/PycharmProjects/cross_validation_paper'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
batch_size = 512
torch.manual_seed(33)
np.random.seed(33)

#load the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
mnist_60000 = datasets.MNIST(root=os.path.join(parent_dir, 'data'), transform = transform, train=True, download=True)
mnist_10000 = datasets.MNIST(root=os.path.join(parent_dir, 'data'),  transform = transform, train=False, download=True)
mnist_70000 = torch.utils.data.ConcatDataset((mnist_10000, mnist_60000))
#create datasets with 5 random seeds
seeds = [11, 22, 33, 44, 55]
mnist_500 = []
mnist_1000 = []
mnist_10000 = []
mnist_test = []
for seed in seeds:
  temp_mnist_10000, temp_mnist_test = torch.utils.data.random_split(mnist_70000, [10000, 60000], generator=torch.Generator().manual_seed(seed))
  temp_mnist_500, _ = torch.utils.data.random_split(temp_mnist_10000, [500, 9500], generator=torch.Generator().manual_seed(seed))
  temp_mnist_1000, _ = torch.utils.data.random_split(temp_mnist_10000, [1000, 9000], generator=torch.Generator().manual_seed(seed))
  mnist_500.append(temp_mnist_500)
  mnist_1000.append(temp_mnist_1000)
  mnist_10000.append(temp_mnist_10000)
  mnist_test.append(temp_mnist_test)



# run through 5-fold CV, for different model types, diffferent epochs, save to csv
model_save_directory = os.path.join(parent_dir, 'model', '5-fold-cv')
epoch_checkpionts = [20, 100]
model_type = ['ConvModel', 'NNModel']
f = open(os.path.join(parent_dir, 'results', 'mnist500_cv.csv'), 'w')
writer = csv.writer(f)

for i, seed in enumerate(seeds):
    dataset = mnist_500[i]
    model_save_path = os.path.join(model_save_directory, 'mnist_500', 'seed' + str(seed))

    # #k-fold CV
    paths, acc = kfoldcv(dataset, 5, model_type, epoch_checkpionts, model_save_path, batch_size, device, criterion)
    writer.writerow(paths)
    writer.writerow(acc)

    # now train model on full dataset (not folds) for comparision with eval
    full_non_cv_loader = DataLoader(dataset, batch_size=batch_size)
    model_paths = []
    model_acc = []
    for model_i in model_type:
        print('Model: {}, full'.format(model_i))
        model = select_model(model_i)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=.0005)
        save_paths, val_acc_at_checkpoint = train_model(epoch_checkpoints=epoch_checkpionts,
                                                        # save model at each checkpoint
                                                        train_loader=full_non_cv_loader,
                                                        valid_loader=None,
                                                        model=model,
                                                        optimizer=optimizer,
                                                        device=device,
                                                        save_path=os.path.join(model_save_path, model_i, 'full'),
                                                        # folder path, not filename
                                                        criterion=criterion)
        model_paths.append(save_paths)
        model_acc.append(val_acc_at_checkpoint)
    writer.writerow(model_paths)
    writer.writerow(model_acc)

f.close()

# Evaluate on test dataset
model_load_directory = model_save_directory
epoch_checkpionts = [20, 100]
model_type = ['ConvModel', 'NNModel']
f2 = open(os.path.join(parent_dir, 'results', 'mnist500_eval.csv'), 'w')
writer = csv.writer(f2)

models_acc = []
models_paths = []
for i, seed in enumerate(seeds):
    dataset = mnist_test[i]
    data_loader = DataLoader(dataset, batch_size=batch_size)
    model_save_path = os.path.join(model_load_directory, 'mnist_500', 'seed' + str(seed))
    # get k-folds
    for fold in range(1, 6):
        for model_i in model_type:
            for epoch_c in epoch_checkpionts:
                model_eval = None
                model_eval = select_model(model_i)
                model_path_j = os.path.join(model_load_directory, 'mnist_500', 'seed' + str(seed), model_i,
                                            'fold' + str(fold), 'model_' + str(epoch_c) + 'e.pt')
                print(model_path_j)
                checkpoint = torch.load(model_path_j)
                model_eval.load_state_dict(checkpoint['model_state_dict'])
                model_eval.eval()
                acc = test_model(loader=data_loader, model=model_eval)
                print(acc)
                models_paths.append(model_path_j)
                models_acc.append(acc)

    # get full (no folds)
    for model_i in model_type:
        for epoch_c in epoch_checkpionts:
            model = select_model(model_i)
            model_path_j = os.path.join(model_load_directory, 'mnist_500', 'seed' + str(seed), model_i, 'full',
                                        'model_' + str(epoch_c) + 'e.pt')
            print(model_path_j)
            checkpoint = torch.load(model_path_j)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            acc = test_model(loader=data_loader, model=model)
            print(acc)
            models_paths.append(model_path_j)
            models_acc.append(acc)

writer.writerow(models_paths)
writer.writerow(models_acc)
f2.close()