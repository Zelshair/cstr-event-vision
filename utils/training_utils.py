import torch
from torch.utils.data.dataloader import DataLoader
from models.classification_model import ClassificationModel
from torch.optim import Adam, SGD
import math
import time
import os
import psutil

# Helper function to set dataloaders for training and evaluation
def set_dataloaders(config, train_dataset, val_dataset, test_dataset):
    # create dataloader for training, validation, and testing sets
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=False,\
                                  generator=torch.Generator().manual_seed(config.SEED), persistent_workers= config.persistent_train_workers, worker_init_fn=worker_init_fn)
    
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.NUM_WORKERS * (not config.cache_val_set),\
                                pin_memory=False, persistent_workers=config.persistent_val_workers, worker_init_fn=worker_init_fn)
        
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.NUM_TEST_WORKERS, pin_memory=False, \
                                 persistent_workers=False, worker_init_fn=worker_init_fn)


    return train_dataloader, val_dataloader, test_dataloader


def worker_init_fn(worker_id):
        """
        Initialize each worker by setting the CPU affinity based on its worker ID.

        Args:
        - worker_id (int): process ID of the initiated worker.
        """
        core_id = worker_id % psutil.cpu_count()
        p = psutil.Process(os.getpid())
        p.cpu_affinity([core_id])
        # print(f"Initiating worker id {worker_id} on core {core_id}")       

# Create Object Classification/Recognition Model based on the specified configuration
def get_classification_model(config, num_classes, size):
    return ClassificationModel(num_classes, config.channels, width=size[0], height=size[1], classifier=config.classifier, pretrained=config.pretrained_classifier)


# helper function to set optimizer based on the specified configuration
def set_optimizer(config, model):
    # Supported optimizers: [ADAM, SGD] 
    # NOTE: add other optimizers as desired

    if config.optimizer == 'adam':
        return Adam(model.parameters(), lr=config.INIT_LR, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        return SGD(model.parameters(), lr=config.INIT_LR, momentum=0)
    # return ADAM by default
    else:
        return Adam(model.parameters(), lr=config.INIT_LR)

def set_scheduler(config, optimizer, VERBOSE=True):
    # Set scheduler if configuration parameter is set
    if config.scheduler:
        if config.scheduler.startswith('Linear'):
            return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=config.INIT_LR, end_factor=config.INIT_LR/10, total_iters=config.EPOCHS, verbose=VERBOSE)
        
        if config.scheduler.startswith('Exponential'):
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=VERBOSE)
        
        elif config.scheduler.startswith('Step'):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, verbose=VERBOSE)
        
        elif config.scheduler.startswith('CosineAnnealing'):
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, verbose=VERBOSE)
        
        elif config.scheduler.startswith('Plateau'):
            if config.use_mAP_val:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2, verbose=VERBOSE)
            else:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=VERBOSE)
        
        elif config.scheduler.startswith('OneCycle'):
            # To be implemented
            # return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.INIT_LR, epochs=config.EPOCHS, verbose=VERBOSE)
            return None
        # Return None if scheduler is not supported
        else:
            return None
    # Return None if scheduler is not set
    else:
        return None

# helper function that abstracts the model training loop
def train_model(model, train_dataloader, val_dataloader, config, optimizer, loss_function, scheduler, VERBOSE=True):
    # initialize a dictionary to store training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # initialize best model state
    best_model_state = model.state_dict()

    # calculate steps per epoch for training and validation set
    train_steps = len(train_dataloader.dataset) / config.batch_size
    val_steps = len(val_dataloader.dataset) / config.batch_size    

    # Keep track of the best validation performance and the number of epochs without improvement
    best_val_loss = torch.tensor(float('inf'))
    no_improvement_epochs = 0    

    # measure how long training is going to take
    print("[INFO] training the network...")
    start_time = time.time()

    # loop over our epochs
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()

        print("[INFO] EPOCH: {}/{}".format(epoch + 1, config.EPOCHS))

        # train model for one epoch        
        total_train_loss, train_correct = train_one_epoch(config, model, train_dataloader, train_steps, optimizer, loss_function)

        # validate model and return validation loss and number of correct predictions
        total_val_loss, val_correct = validate(config, model, val_dataloader, val_steps, loss_function)

        # Record epoch end time
        epoch_end_time = time.time()

        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps

        # calculate the training and validation accuracy
        train_correct = train_correct / len(train_dataloader.dataset)
        val_correct = val_correct / len(val_dataloader.dataset)    

        # print the model training and validation information as well as the elapsed time of the current epoch
        if VERBOSE:
            print("- Train loss: {:.6f}, Train accuracy: {:.4f}".format(avg_train_loss, train_correct))
            print("- Val loss: {:.6f}, Val accuracy: {:.4f}".format(avg_val_loss, val_correct))
            print("Epoch time: {:.2f} s".format(epoch_end_time - epoch_start_time))

        # update our training history
        history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
        history["train_acc"].append(train_correct)
        history["val_loss"].append(avg_val_loss.cpu().detach().numpy())
        history["val_acc"].append(val_correct)

        # Save the model parameters if the validation loss improves
        if avg_val_loss.cpu().detach() < best_val_loss:
            print("Model validation loss improved from", best_val_loss.detach().cpu().numpy(), \
                "to", avg_val_loss.detach().cpu().numpy(), '\n')
            best_val_loss = avg_val_loss

            if config.save_results:
                # Create output directory if it does not exist
                os.makedirs(os.path.join(config.OUTPUT_PATH, config.CKPT_PATH), exist_ok=True)
                # save model to file
                torch.save(model.state_dict(), os.path.join(config.OUTPUT_PATH, config.CKPT_PATH, config.best_model))
            
            # save best model state in memory
            best_model_state = model.state_dict()
            # reset no improvement counter
            no_improvement_epochs = 0

        else:
            no_improvement_epochs += 1
            print("Model did not improve from a validation loss of", best_val_loss.detach().cpu().numpy(), '\n')

        # Stop the training process if the validation loss has not improved for a specified number of epochs
        if no_improvement_epochs >= config.EARLY_STOP_THRESH:
            print("No validation loss improvement for {} epochs. Early stopping...".format(str(no_improvement_epochs)), '\n')
            break

        # Step the scheduler after each epoch (if utilized)
        if scheduler:
            # Step the ReduceOnPlateau scheduler after each epoch (if utilized)
            if config.scheduler.startswith('Plateau'):
                scheduler.step(avg_val_loss)
            # Step scheduler
            else:
                scheduler.step()

    # finish measuring how long training took
    end_time = time.time()

    if VERBOSE:
        print("[INFO] total time taken to train the model: {:.2f} s".format(end_time - start_time))
        print("[INFO] loading best model...")
    
    # load best model from stored state_dict in memory
    model.load_state_dict(best_model_state)

    return model, history

# helper function to train a classification model for one epoch
def train_one_epoch(config, model, train_dataloader, train_steps, optimizer, loss_function, VERBOSE=True):
    # set the model in training mode
    model.train()

    # initialize the total training loss
    total_train_loss = 0
    
    # initialize the number of correct predictions in the training step
    train_correct = 0

    # initialize step counter
    step_no = 0

    # loop over the training set
    for (event_frame, label) in train_dataloader:
        # send the input to the GPU/designated Device
        (event_frame, label) = (event_frame.to(config.DEVICE), (label.to(config.DEVICE)))    

        # check if classifier is inceptionv3 with Aux logits enabled
        if config.classifier == 'inceptionv3_aux':
            # verify that the model's aux_logits are enabled
            assert model.classifier.aux_logits == True

            # perform a forward pass and calculate the training loss with aux_logits
            pred, aux_pred = model(event_frame)
            loss1 = loss_function(pred, label)
            loss2 = loss_function(aux_pred, label)
            loss = loss1 + 0.4*loss2

        else:
            # perform a forward pass and calculate the training loss
            pred = model(event_frame)
            loss = loss_function(pred, label)
        
        # zero out the gradients, perform backpropagation step, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add current loss to total training loss
        total_train_loss += loss

        # calculate correct predicts per current training step
        train_step_correct = (pred.argmax(1) == label).type(torch.float).sum().item()
        # add correct predictions to total correct predictions counter
        train_correct += train_step_correct
        # increment step number
        step_no += 1
        
        if VERBOSE:
            print("train step [{}/{}] - loss: {:.6f}, accuracy: {:.4f}".format(step_no, math.ceil(train_steps), loss, train_step_correct/label.shape[0]), end='\r')
    
    if VERBOSE:
        print()

    # return total training loss and number of correct predictions
    return total_train_loss, train_correct

# helper function to validate model after training for one epoch
def validate(config, model, val_dataloader, val_steps, loss_function, VERBOSE=True):
    # switch off autograd for evaluation
    with torch.no_grad():
        # set the model in evaluation mode        
        model.eval()

        # initialize the total validation loss
        total_val_loss = 0

        # initialize the number of correct predictions counter
        val_correct = 0

        # initialize step counter
        val_step_no = 0

        # loop over the validation set
        for(event_frame, label) in val_dataloader:
            # send the input to the GPU/designated Device
            (event_frame, label) = (event_frame.to(config.DEVICE)), label.to(config.DEVICE)    

            # make the predictions and calculate the validation loss
            pred = model(event_frame)
            val_loss = loss_function(pred, label)

            total_val_loss += val_loss

            # calculate the number of correct predictions
            val_step_correct = (pred.argmax(1) == label).type(torch.float).sum().item()
            val_correct += val_step_correct

            val_step_no += 1

            if VERBOSE:
                print("val step [{}/{}] - val loss: {:.6f}, val accuracy: {:.4f}".format(val_step_no, math.ceil(val_steps), val_loss, val_step_correct/label.shape[0]), end='\r')
    
        if VERBOSE:
            print()
        
        # return total validation loss and number of correct predictions
        return total_val_loss, val_correct

# helper function to load the best model checkpoint either from file or memory
def load_best_model(model, config, best_model_state = None, VERBOSE=True):
    if VERBOSE:
        print("[INFO] Loading best model from checkpoint")

    # load best checkpoint weights from file if saving is enabled
    if config.save_results:
        model.load_state_dict(torch.load(os.path.join(config.OUTPUT_PATH, config.MODELS_PATH, config.best_model)))
    else:
        assert best_model_state is not None, "if saving/reading from file is not enabled, you must provide a state_dict for the best trained model!"
        model.load_state_dict(best_model_state)

# helper function to save the best model to file if enabled
def save_model(config, model):
    if config.save_results:
        # Create output directory if it does not exist
        os.makedirs(os.path.join(config.OUTPUT_PATH, config.MODELS_PATH), exist_ok=True)

        # serialize the model to disk
        torch.save(model, os.path.join(config.OUTPUT_PATH, config.MODELS_PATH, config.best_model))

# helper function to print training/testing configuration
def print_configuration(config, dataset):
    print("[Training configuration]:")
    print("\tEvent dataset =", dataset)
    print("\tDataset input spatial dimensions = ({} x {})".format(dataset.width, dataset.height))

    if config.keep_size:
        print("\tModel input size = ({}, {}, {})".format(config.channels, dataset.width, dataset.width))
    else:
        print("\tModel input size = ({}, {}, {})".format(config.channels, config.frame_size, config.frame_size))

    print("\tNumber of classes =", dataset.num_classes)
    print("\tClassifier =", config.classifier)
    print("\tEvent Representation =", config.event_rep)
    print("\tUse pretrained weights =", config.pretrained_classifier)
    print('\tBalanced splits =', config.balanced_splits)
    
    if config.augmentation:
        config.augmentation_config.print_config()

    if config.delta_t > 0:
        print('\tDelta_t (event sampling duration) =', config.delta_t, 'ms')

    print('\tBatch Size =', config.batch_size)
    print('\tOptimizer =', config.optimizer)
    print('\tLearning Rate =', config.INIT_LR)
    print('\tWeight decay =', config.weight_decay)

    if config.scheduler:
        print('\tScheduler =', config.scheduler)

    if config.normalization is None:
        print("\tNormalization = None")
    else:
        print("\tNormalization =", config.normalization)
        print("\t\tmean =", config.MEAN)
        print("\t\tstd =", config.STD)

    print()

# Helper function to set PyTorch seeds.
def set_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    