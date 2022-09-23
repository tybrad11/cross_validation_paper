import torch
import numpy as np
import os
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader,SubsetRandomSampler
from models import select_model


def train_model(epoch_checkpoints=[5, 15, 30, 100],  # save model at each checkpoint
                train_loader=None,
                valid_loader=None,
                model=None,
                optimizer=None,
                device=None,
                save_path=None,  # folder path, not filename. Leave None to not save
                criterion=None):  # loss function
  val_acc_at_checkpoint = []
  saved_models = []

  if not os.path.exists(save_path):
    os.makedirs(save_path)
  for epoch in range(1, max(epoch_checkpoints) + 1):
    # puts the model into train model so it updates weights
    model.train()

    loss_array = []

    for images, targets in train_loader:
      targets = targets.to(device, dtype=torch.long)
      images = images.to(device)

      # the prediction for the model
      outputs = model(images)

      # gets the loss based on the model predictions
      loss = criterion(outputs, targets)
      loss_array.append(loss.cpu().detach().numpy().tolist())

      # zeros out the gradient from last step
      optimizer.zero_grad()
      # propagates loss to all the weights in the model
      loss.backward()
      # takes the step given the propagated loss and the optimzer
      optimizer.step()
    if epoch % 5 == 0:
      # should make average loss
      print(f"Epoch {str(epoch)}, Average Training Loss = {np.mean(np.asarray(loss_array))}")

    #
    if epoch in epoch_checkpoints:
      if save_path != None:
        save_name = os.path.join(save_path, 'model_' + str(epoch) + 'e.pt')
        saved_models.append(save_name)
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss, }, save_name)
        print('Saved {}'.format(save_name))
      else:
        saved_models = ''

      if valid_loader is not None:
        validation_acc = test_model(loader=valid_loader, model=model, device=device)
        print("   Evaluation accuracy for epoch {} : {}".format(str(epoch), str(validation_acc)))
        val_acc_at_checkpoint.append(validation_acc)
  return saved_models, val_acc_at_checkpoint


def test_model(loader=None, model=None, device=None):
  # this should make sure the weights don't require a gradient
  model.eval()
  fin_targets = []
  fin_outputs = []
  # loops through all the batches of images and labels in our test set
  for images, targets in loader:
    targets = targets.to(device, dtype=torch.long)
    images = images.to(device)

    # the prediction for the model
    outputs = model(images)
    # the model makes a prediction
    outputs = model(images)
    # the outputs are convered to lists appened onto out final prediction list
    fin_targets.extend(targets.cpu().detach().numpy().tolist())
    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

  # maybe cut this line
  final_outputs = np.copy(fin_outputs)
  # get the index of the best prediction
  final_outputs = np.argmax(final_outputs, axis=1)

  # gets the number of correct predictions from our final output
  num_correct = 0
  for i in range(0, len(fin_targets)):
    if (fin_targets[i] == final_outputs[i]):
      num_correct = num_correct + 1

  # calculates and returns the accuracy on the given test set
  accuracy = num_correct / len(fin_targets)
  return accuracy



def kfoldcv(dataset, n_folds, model_type, epoch_checkpionts, model_save_path, batch_size, device, criterion):
  splits = KFold(n_splits = n_folds, shuffle=True, random_state=33)
  model_paths = []
  model_acc = []
  for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('------------ Fold {} --------------'.format(str(fold + 1)))
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    for model_i in model_type:
      print('Model: {}, fold {}'.format(model_i, str(fold+1)))
      model = select_model(model_i)
      model.to(device)
      optimizer = torch.optim.Adam(params=model.parameters(), lr=.0005)
      save_paths, val_acc_at_checkpoint = train_model(epoch_checkpoints = epoch_checkpionts,  #save model at each checkpoint
                    train_loader = train_loader,
                    valid_loader = val_loader,
                    model= model,
                    optimizer= optimizer,
                    device = device,
                    save_path = os.path.join(model_save_path, model_i, 'fold' + str(fold+1)),  #folder path, not filename
                    criterion = criterion)
      model_paths.append(save_paths)
      model_acc.append(val_acc_at_checkpoint)
  return model_paths, model_acc