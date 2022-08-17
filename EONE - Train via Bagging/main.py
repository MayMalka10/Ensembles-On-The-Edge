import os
import sys
import torch
import hydra
import ssl
import torch.nn as nn
from torchvision import datasets
from omegaconf import DictConfig
import numpy as np
from Utils.transforms import get_cifar_train_transforms, get_test_transform
from Archs.MobileNetV2 import SplitEffNet
from ColabInferModel import NeuraQuantModel, NeuraQuantModel2
import time
import pickle

is_debug = sys.gettrace() is not None

@hydra.main(config_path="Config_Files", config_name="config")
def train(cfg: DictConfig) -> None:

    ## creates headline with the current data ###
    log_dir_name = f"{cfg.Dataset.Dataset.name}_{cfg.Architecture.Architecture.name}_" \
                   f"{cfg.Quantization.Quantization}_{cfg.Ensemble.Ensemble}" \
                    .replace(":", '-').replace('\'', '').replace(' ', '')
    log_dir_name = log_dir_name.replace('{', '')
    log_dir_name = log_dir_name.replace('}', '')
    log_dir_name = log_dir_name.replace(',', '_')

    ### checking if in debug mode for much lighter network ###
    if is_debug:
        print('in debug mode!')
        training_params = cfg.Training.training_debug
    else:
        print('in run mode!')
        training_params = cfg.Training.Training

    ### Prepare the Data ###

    test_transform = get_test_transform()
    if 'cifar' in cfg.Dataset.Dataset['name']:
        train_transform = get_cifar_train_transforms()
    else:
        train_transform = test_transform

    if cfg.Dataset.Dataset['name'] == 'cifar10':
        ssl._create_default_https_context = ssl._create_unverified_context
        trainset = datasets.CIFAR10(root=cfg.params.data_path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root=cfg.params.data_path, train=False, download=True, transform=test_transform)


    elif cfg.Dataset.Dataset['name'] == 'cifar100':
        ssl._create_default_https_context = ssl._create_unverified_context
        trainset = datasets.CIFAR100(root=cfg.params.data_path, train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR100(root=cfg.params.data_path, train=False, download=True, transform=test_transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=training_params.batch_size,
                                              shuffle=True, num_workers=training_params.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=training_params.batch_size,
                                             shuffle=False, num_workers=training_params.num_workers)
    if cfg.Architecture.Architecture.name == 'mobilenet':
        EncDec_dict = SplitEffNet(width=cfg.Architecture.Architecture.width, pretrained=True,
                                     num_classes=cfg.Dataset.Dataset.num_classes,
                                     stop_layer=cfg.Quantization.Quantization.part_idx,
                                     decoder_copies=cfg.Ensemble.Ensemble.n_ensemble,
                                     architectura=cfg.Architecture.Architecture.architectura,
                                     pre_train_cifar=cfg.Architecture.Architecture.pre_train_cifar)


    path_to_pretrained_dict = os.getcwd()[0:-27] + 'Saves/EncDec_dict_test.pkl'
    if cfg.Training.Training.un_quantized_training:
        with open('EncDec_dict_test.pkl', 'wb') as f:
            pickle.dump(EncDec_dict, f)

    if not cfg.Training.Training.un_quantized_training:
        with open(path_to_pretrained_dict, 'rb') as f:
            EncDec_dict = pickle.load(f)

    learning_rate = cfg.Training.Training.lr
    criterion = nn.CrossEntropyLoss()
    dec = EncDec_dict['decoders'][0]

    model = NeuraQuantModel(encoder=EncDec_dict['encoder'],
                            decoder=[dec],
                            primary_loss=criterion,
                            n_embed=cfg.Quantization.Quantization.n_embed,
                            n_parts=cfg.Quantization.Quantization.n_parts,
                            commitment=cfg.Quantization.Quantization.commitment_w)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    start_time = time.time()
    train_losses = []
    test_losses = []
    epoch_acc = []


    num_users = cfg.Ensemble.Ensemble.n_ensemble
    num_classes = cfg.Dataset.Dataset.num_classes
    epochs = 1 #set number of epochs



    entries = len(testset) // training_params.batch_size
    train_first_acc_matrix = torch.empty([epochs, 1])
    test_first_acc_matrix = torch.empty([epochs, 1])
    y_hat_tensor = torch.empty([num_users, epochs, entries, training_params.batch_size, num_classes])

    # Number of parameters
    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    print(f'\nNumber of Parameters: {get_n_params(model)}\n')
    subset = 34000
    start_idx = 34000
    stop_idx = 35000

    # if cfg.Ensemble.Ensemble.n_ensemble > 1:
    #     numbers = np.arange(0,subset)
    #     rand_ints = list(np.sort(np.unique(np.random.choice(numbers, dense))))
    #     smallset = torch.utils.data.Subset(trainset, rand_ints)
    #     trainloader_first = torch.utils.data.DataLoader(smallset, batch_size=training_params.batch_size,
    #                                               shuffle=True, num_workers=training_params.num_workers)


    shared_idxs = np.arange(0,subset)
    first = np.arange(start_idx,stop_idx)
    start_idx = stop_idx
    stop_idx = stop_idx+1000
    first_model_indices = np.concatenate((shared_idxs,first))
    smallset = torch.utils.data.Subset(trainset, first_model_indices)
    trainloader_first = torch.utils.data.DataLoader(smallset, batch_size=training_params.batch_size,
                                                   shuffle=True, num_workers=training_params.num_workers)



    ###############################
    ##### Train First Model #######
    ###############################

    if cfg.Training.Training.train_first:

        for epc in range(epochs):
            bingos = 0
            losses = 0
            for batch_num, (Train, Labels) in enumerate(trainloader_first):
                batch_num += 1
                if len(Train) != training_params.batch_size:
                    continue
                batch = (Train, Labels)
                result_dict, batch_acc, y_hat = model.process_batch(batch)
                loss = result_dict['loss']
                losses += loss.item()
                bingos += batch_acc[0]


                # Update parameters
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch_num % 32 == 0:
                    print(f'epoch: {epc:2}  batch: {batch_num:2} [{training_params.batch_size * batch_num:6}/{len(smallset)}]  total loss: {loss.item():10.8f}  \
                    time = [{(time.time() - start_time)/60}] minutes')

            scheduler.step()
            ### Accuracy ###
            loss = losses/batch_num
            # writer.add_scalar("Train Loss (Model1)", loss, epc)
            train_losses.append(loss)
            train_losses.append(loss)
            accuracy = 100 * bingos.item() / len(smallset)
            epoch_acc.append(accuracy)
            # writer.add_scalar("Train Accuracy (Model1)", accuracy, epc)
            train_first_acc_matrix[epc][0] = epoch_acc[epc]

            num_val_correct=0
            model.eval()
            test_losses_val = 0

            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(testloader):
                    if len(X_test) != training_params.batch_size:
                        continue
                    # Apply the model
                    b += 1
                    batch = (X_test, y_test)
                    result_dict, test_batch_acc, y_hat = model.process_batch(batch)
                    test_loss = result_dict['loss']
                    test_losses_val += test_loss.item()
                    num_val_correct += test_batch_acc[0]
                    y_hat_tensor[0, epc, b - 1, :, :] = y_hat[0]

                test_losses.append(test_losses_val / b)
                # writer.add_scalar("Test Loss (Model1)", test_losses_val / b, epc)
                test_first_acc_matrix[epc][0] = 100*num_val_correct/len(testset)
                # writer.add_scalar("Test Accuracy (Model1)", num_val_correct / 100, epc)

            print(f'Train (Model1) Accuracy at epoch {epc + 1} is {100*bingos.item()/len(smallset)}%')
            print(f'Validation (Model1) Accuracy at epoch {epc+1} is {100*num_val_correct/len(testset)}%')
            model.train()

        # torch.save(model.encoder.state_dict(), 'NeuraQuantizerEncoder.pt')
        # torch.save(model.quantizer.state_dict(), 'NeuraQuantizerQuantizer.pt')



    ###############################
    ##### Train Next Models #######
    ###############################


    train_rest_acc_matrix = torch.empty([epochs, num_users-1])
    test_rest_acc_matrix = torch.empty([epochs, num_users-1])
    # y_hat_tensor_rest = torch.empty([num_users-1, epochs, entries, training_params.batch_size, num_classes])


    for num in range(1,len(EncDec_dict['decoders'])):
        # strings for tensorboard
        model_name = 'Model' + str(num + 1)

        dec = EncDec_dict['decoders'][num]
        criterion2 = nn.CrossEntropyLoss()
        model2 = NeuraQuantModel2(encoder=model.encoder.train(False),
                                  decoder=[dec],
                                  quantizer=model.quantizer.train(False),
                                  primary_loss=criterion2,
                                  n_embed=cfg.Quantization.Quantization.n_embed,
                                  commitment=cfg.Quantization.Quantization.commitment_w)

        model2.encoder.eval()
        for param in model2.encoder.parameters():
            param.requires_grad = False


        optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=3, gamma=0.9)
        next = np.arange(start_idx, stop_idx)
        start_idx = stop_idx
        stop_idx = stop_idx + 1000
        next_model_indices = np.concatenate((shared_idxs, next))
        smallset1 = torch.utils.data.Subset(trainset, next_model_indices)
        trainloader_next = torch.utils.data.DataLoader(smallset1, batch_size=training_params.batch_size,
                                                   shuffle=True, num_workers=training_params.num_workers)

        for epc in range(epochs):
            bingos = 0
            losses = 0

            for batch_num, (Train, Labels) in enumerate(trainloader_next):
                if len(Train) != training_params.batch_size:
                    continue
                batch_num += 1
                batch = (Train, Labels)
                result_dict, batch_acc, y_hat = model2.process_batch_fixed(batch)
                loss = result_dict['loss']
                losses += loss.item()
                bingos += batch_acc[0]

                # Update parameters
                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()

                if batch_num % 32 == 0:
                    print(
                        f'epoch: {epc:2}  batch: {batch_num:2} [{training_params.batch_size * batch_num:6}/{len(smallset1)}]  total loss: {loss.item():10.8f}  \
                    time = [{(time.time() - start_time) / 60}] minutes')

            scheduler2.step()
            ### Accuracy ###
            tot_loss = losses / batch_num
            train_losses.append(tot_loss)
            # writer.add_scalar(train_loss_str, tot_loss, epc)
            accuracy = 100*bingos.item()/len(smallset1)
            # writer.add_scalar(train_acc_str, accuracy, epc)
            epoch_acc.append(accuracy)
            train_rest_acc_matrix[epc][num-1] = epoch_acc[epc]

            num_val_correct = 0
            model2.decoder.eval()
            test_losses_val = 0

            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(testloader):
                    if len(X_test) != training_params.batch_size:
                        continue
                    # Apply the model
                    b += 1
                    batch = (X_test, y_test)
                    result_dict, test_batch_acc, y_hat = model2.process_batch_fixed(batch)
                    test_loss = result_dict['loss']
                    test_losses_val += test_loss.item()
                    num_val_correct += test_batch_acc[0]
                    y_hat_tensor[num, epc, b - 1, :, :] = y_hat[0]

                test_losses.append(test_losses_val / b)
                # writer.add_scalar(test_loss_str, test_losses_val / b, epc)
                test_rest_acc_matrix[epc][num-1] = 100*num_val_correct/len(testset)
                # writer.add_scalar(test_acc_str, num_val_correct / 100, epc)



            print(f'Train {model_name} Accuracy at epoch {epc + 1} is {100*bingos/len(smallset1)}%')
            print(f'Validation {model_name} Accuracy at epoch {epc + 1} is {100*num_val_correct/len(testset)}%')
            model2.decoder.train()

        ##### Calculate the Ensemble Validation Accuracy ######

    torch.save(y_hat_tensor, 'y_hat_tensor.pt')

    ### Save pretrained model with no quantization
    if cfg.Training.Training.un_quantized_training:
        with open('EncDec_dict.pkl', 'wb') as f:
            pickle.dump(EncDec_dict, f)

    def ensemble_calculator(preds_list, num_users):
        stack = preds_list.view([num_users, -1])
        return torch.mean(stack, axis=0)

    def accuracy(y, ensemble_y_pred):
        ens_pred = torch.max(ensemble_y_pred.data, 1)[1]
        batch_ens_corr = (ens_pred == y).sum()
        return batch_ens_corr

    ### Calculate the predictions of the Ensembles

    ensemble_y_hat = torch.empty([num_users, epochs, entries, training_params.batch_size, num_classes])

    for num_of_ens in range(num_users):
        for epc in range(epochs):
            preds = y_hat_tensor[:num_of_ens + 1, epc, :, :, :]
            mean = ensemble_calculator(preds, num_of_ens + 1)
            mean = mean.view([-1, training_params.batch_size, num_classes])
            ensemble_y_hat[num_of_ens, epc, :, :, :] = mean

    ensemble_accuracy_per_users = torch.empty([num_users, epochs, entries])
    accuracy_ensemble_tensor = torch.empty([num_users, epochs])

    ### Checking the accuracy of the ensemble predictions

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(testloader):
            if len(X_test) != training_params.batch_size:
                continue
            for num_of_ens in range(num_users):
                for epc in range(epochs):
                    y_pred = ensemble_y_hat[num_of_ens, epc, b, :, :]
                    batch_ens_corr = accuracy(y_test, y_pred)
                    ensemble_accuracy_per_users[num_of_ens, epc, b] = batch_ens_corr

        for num_of_ens in range(num_users):
            for epc in range(epochs):
                total_correct = ensemble_accuracy_per_users[num_of_ens, epc, :]
                sum_of_correct = total_correct.sum()
                acc_correct = sum_of_correct * (100 / 10000)
                accuracy_ensemble_tensor[num_of_ens, epc] = acc_correct

    ## save losses to file
    torch.save(accuracy_ensemble_tensor, 'accuracy_ensemble_tensor.pt')
    torch.save(test_rest_acc_matrix, 'test_rest_acc_matrix.pt')
    torch.save(train_rest_acc_matrix, 'train_rest_acc_matrix.pt')


    for i in range(num_users):
        print(f'Accuracy of Ensemble of {i+1} Models: {accuracy_ensemble_tensor[i, -1].item():.3f}%')

    print('\nTRAINING HAS FINISHED SECCESSFULLY')
    print(log_dir_name)

### Train the Model ###
if __name__ == '__main__':
    train()
