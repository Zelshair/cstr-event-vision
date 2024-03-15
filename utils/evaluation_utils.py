import torch
import math
import time
import os
import cv2

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn.metrics import classification_report

# helper function that abstracts evaluating a model on a test set
def evaluate_model(config, model, test_dataloader, VERBOSE=True):
    if VERBOSE:
        print("[INFO] evaluating network...")

    # Measure how long evaluation is going to take
    start_time = time.time()

    # Calculate the test steps for the test set
    test_steps = math.ceil(len(test_dataloader.dataset) / config.batch_size)    

    # Initialize the number of correct predictions counter
    test_correct = 0

    # Initialize step counter
    test_step_no = 0

    # Set the model in evaluation mode
    model.eval()    
    
    # Turn off autograd for testing evaluation
    with torch.no_grad():        
        # Initialize a list to store our predictions
        preds = []

        # Loop over the test set
        for (event_frame, label) in test_dataloader:
            # send the input to the designated device
            (event_frame, label) = (event_frame.to(config.DEVICE)), label.to(config.DEVICE)   

            # Make the predictions and add them to the list
            pred = model(event_frame)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

            # calculate the number of correct predictions
            test_step_correct = (pred.argmax(1) == label).type(torch.float).sum().item()
            test_correct += test_step_correct

            test_step_no += 1

            if VERBOSE:
                print("test step [{}/{}] - test accuracy: {:.4f}".format(test_step_no, test_steps, test_step_correct/label.shape[0]), end='\r')
            
            # Visualize results if enabled
            if config.visualize_results:
                visualize_classification(config, event_frame, label, pred.argmax(1), test_dataloader.dataset.classes, test_dataloader.dataset.get_frame_dims(), config.norm_params)

        # Finish measuring how long evaluation took
        end_time = time.time()

        if VERBOSE:
            print("\n[INFO] total time taken to evaluate the model: {:.2f}s".format(end_time - start_time))
            print("Test Accuracy = {:.4f}\n".format(test_correct / len(test_dataloader.dataset)))
    
    # Return predictions
    return preds

# Helper function to visualize results and save to file if desired
def visualize_classification(config, frames, labels, preds, class_names, org_dims, normalization):
    assert config.channels in [1,2,3], "Only supports visualization for 1, 2, or 3-Channel representations"
    # if config.channels == 3:
    # Create transformation to remove image normalization previously applied to the input
    denorm = transforms.Compose([transforms.Normalize(mean = [0.]*config.channels,
                                                        std = 1/np.array(normalization.std)),
                                transforms.Normalize(mean = -np.array(normalization.mean),
                                                        std = [ 1.]*config.channels),
                                ])
    
    # Remove normalization from image using transforms
    frames = denorm(frames)

    # Loop over each frame in the batch
    for i, frame in enumerate(frames):
        # Convert to numpy ndarray and move to CPU
        output_frame = frame.cpu().numpy()
        
        # Reoder dimensions from (C, W, H) to (W, H, C)
        output_frame = np.transpose(output_frame, (1, 2, 0)).astype(np.float32)

        # Remove normalization by multiplying by 255 then converting to 'uint8'
        output_frame = (output_frame * 255).astype('uint8')

        if config.channels == 1:
            # Convert from grayscale (1-channel) to BGR for visualization
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_GRAY2BGR)
        
        elif config.channels == 2:
            # Extended output frame to 3 channels and visualize using Red and Blue channels
            extended_frame = np.zeros((output_frame.shape[0], output_frame.shape[1], 3), dtype=np.uint8)
            extended_frame[..., 0] = output_frame[..., 0]
            extended_frame[..., 2] = output_frame[..., 1]
            output_frame = extended_frame
            # Convert from RGB to BGR for visualization
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)            

        elif config.channels == 3:
            # Convert from RGB to BGR for visualization
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        # Convert to original frame resolution and aspect ratio
        output_frame = cv2.resize(output_frame, org_dims)
        
        # [Optional] Enlarge frame for better visualization
        width, height = org_dims
        height *= 3
        width *= 3
        output_frame = cv2.resize(output_frame, (width, height))

        display_width = width + 40 if width < 150 else width
        # Add extra space for text
        display_frame = np.zeros((height + 70, display_width, 3), dtype='uint8')

        display_frame[0:height, 0:width] = output_frame

        # Add text for correct label
        cv2.putText(display_frame, f'Correct: {class_names[labels[i]]}', (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add text for predicted label
        cv2.putText(display_frame, f'Predicted: {class_names[preds[i]]}', (10, height + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Event Representation', display_frame)
        key = cv2.waitKey(0)
        
        # exit if esc is pressed
        if key == 27:
            exit()


# helper function to plot the training and validation (loss + accuracy) history and save to file
def plot_train_hist(history, config, model = None):
    assert history is not None, "No history dictionary-variable was provided"
    # plot the training loss and accuracy
    # plt.style.use("ggplot")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.tight_layout()

    if config.save_results:
        print("saving results to file...")
        # Create output directory if it does not exist
        os.makedirs(os.path.join(config.OUTPUT_PATH, config.HIST_PATH), exist_ok=True)
        
        # save plot to file
        plt.savefig(os.path.join(config.OUTPUT_PATH, config.HIST_PATH, config.best_model.split('.')[0] + '.png'))

        # save history to csv file
        data_output = pd.DataFrame(history)

        data_output.to_csv(os.path.join(config.OUTPUT_PATH, config.HIST_PATH, config.best_model.split('.')[0] + '.csv'))

    plt.close()

# helper function that generates classification report based on the predictions vs ground truth labels for the test set
def gen_classification_report(config, labels, predictions, class_names, model_name = None, VERBOSE=True):
    # generate a classification report
    class_report = classification_report(torch.Tensor(labels).cpu().numpy(), \
        np.array(predictions), target_names=class_names, digits=4)
    
    if VERBOSE:
        print(class_report)
        
    if config.save_results:
        # Create output directory if it does not exist
        os.makedirs(os.path.join(config.OUTPUT_PATH, config.RESULTS_PATH), exist_ok=True)

        # Create text file (using provided model name if provided)
        if model_name:
            f =  open(os.path.join(config.OUTPUT_PATH, config.RESULTS_PATH, model_name.split('.')[0] + '.txt'), 'w')
        else:
            f =  open(os.path.join(config.OUTPUT_PATH, config.RESULTS_PATH, config.best_model.split('.')[0] + '.txt'), 'w')

        f.write(class_report)
        
        # Save predictions to results file if enabled (truncated after classification report)
        if config.save_preds:
            f.write('\n')

            for prediction in predictions:
                line = "{}\n".format(class_names[prediction])
                f.write(line)         

        f.close()