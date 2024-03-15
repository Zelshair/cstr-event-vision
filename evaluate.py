"""
Evaluation Script for Pretrained Event-Based Classification/Recognition Models.

This script is designed to evaluate a pretrained model on a specified test dataset. It loads the model and the dataset, performs the evaluation, and generates a classification report. The script allows for visualization of results if configured.

The evaluation process involves:
- Setting up the configuration for the evaluation, including dataset details and model specifics.
- Loading the test dataset using the specified configurations.
- Preparing a DataLoader for batch processing of test data.
- Loading the pretrained model from a specified file path.
- Running the evaluation to generate predictions on the test dataset.
- Generating and optionally saving a classification report summarizing the model's performance.

Usage:
    Run the script directly from the command line. Modify the `config` instantiation in the script to match the desired evaluation setup and ensure the model file name and path are correctly specified.

Example:
    ```
    python evaluate.py
    ```

Ensure the model to be evaluated and the dataset for testing are correctly specified in the script and accessible at the paths defined.

Note:
    - The script assumes the model file is stored in the 'output/models' directory.
    - Visualization options and saving results can be toggled via the `config` object.
"""


import os
from utils import *

def main():
    # set configs
    config = Configs(event_rep='cstr', channels=3, classifier='mobilenetv2', dataset='N-Cars', save_results=True, visualize_results=False)

    # Set paths based on the specified dataset
    dataset_root = dataset_config[config.dataset]['root']
    test_csv_file = dataset_config[config.dataset]['test_csv']
    
    # Get path to test set's ledger
    test_csv_path = os.path.join(dataset_root, test_csv_file)

    # Load the test set
    test_dataset = create_dataset(config, dataset_root, test_csv_path, 'test', transform=set_transforms(config, False))

    # Create Dataloader for the test set
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.NUM_TEST_WORKERS, pin_memory=False)
    # test_dataloader = DataLoader(test_set, batch_size=1, num_workers=0, pin_memory=False) # batch size of 1 is used for visualization

    # Load trained model from file (expected in output/models directory)
    model_name = 'N-Cars_cstr_mobilenetv2_3C_best_model_pretrained_224_aug_STP-ImageNet.pth'
    model_path = os.path.join(config.OUTPUT_PATH, config.MODELS_PATH, model_name)
    
    # Load trained object detection model from file
    test_model = torch.load(model_path).to(config.DEVICE)
    
    # Load_model_weights(test_model, model_path)
    predictions = evaluate_model(config, test_model, test_dataloader)

    # Generate test results and/or save to file
    gen_classification_report(config, test_dataset.class_index, predictions, test_dataset.classes, model_name=model_name)

if __name__ == "__main__":
    main()