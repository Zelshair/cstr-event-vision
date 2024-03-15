""" Split N-Caltech101 / CIFAR10-DVS / ASL-DVS ledger script

This is a script to split N-Caltech101/CIFAR10-DVS/ASL-DVS dataset to a train and test csv files. This can be used for any other datasets without an official split.

Choosing a minimum number of samples per class, ensures that there is sufficient representation of each class in the test (compared to random split of the whole dataset).

Operation info:
- The dataset ledger csv file is initially loaded, then random N samples of each class is extracted and saved to a separate csv file.

- The original one (excluding the N sampled datapoints of each class) is saved to a train csv file.

- Outputs 2 csv files (train & test splits).

"""

import pandas as pd
import os

SEED = 42

def main():
    # select split mode
    mode = 2

    # set number of samples per class
    num_samples = 200

    # set percentage % of each class or the dataset
    pct_split = 0.2


    # set path and file name of the dataset's ledger csv file
    dataset_csv_file =  '../Datasets/CIFAR10-DVS/CIFAR10-DVS.csv'
                    #    '../Datasets/N-Caltech101/Caltech101'
    # split dataset based on number of samples per class specified and save resulting csv files for each
    split_dataset(dataset_csv_file, num_samples, pct_split, mode=mode)


# by random % split per class
# function to split a dataset's ledger csv file into two based on the desired split mode
def split_dataset(dataset_csv, num_samples=15, pct_split = 0.2, mode=0):
    """split data based on the desired split mode:
        mode 0 - random dataset split: take a % split randomly across the whole dataset
        mode 1 - fixed_no_samples: extract a fixed number of samples for each class of the dataset randomly
        mode 2 - class split %: take a % split of each class's samples randomly"""

    # ensure that path is specified in Linux format ('/')
    dataset_csv = dataset_csv.replace('\\', '/')
    # extract path excluding filename
    dataset_path = '/'.join(dataset_csv.split('/')[0:-1])

    # generate train and test output file names
    train_csv_file = dataset_csv.split('.')[-2].split('/')[-1] + '_train_' + str(mode)+ '.csv'
    test_csv_file = dataset_csv.split('.')[-2].split('/')[-1]  + '_test_' + str(mode)+ '.csv'

    # generate output path + filenames
    train_csv_file = os.path.join(dataset_path, train_csv_file)
    test_csv_file = os.path.join(dataset_path, test_csv_file)
    
    # read csv file into pandas dataframe
    dataset_ledger = pd.read_csv(dataset_csv)

    # create an empty list to store the sampled data
    sampled_data = []    

    # create a new DataFrame to store the remaining data
    # remaining_data = pd.DataFrame(columns=['events_file_path', 'class_index'], index=[0])
    remaining_data = pd.DataFrame()

    # get total number of dataset samples
    total_samples = len(dataset_ledger)

    # split the whole dataset randomly based on the split % specified 
    if mode == 0:
        # split dataset randomly based on the defined split %
        sampled_data = dataset_ledger.sample(n=round(pct_split * total_samples), random_state=SEED)
        remaining = dataset_ledger.drop(sampled_data.index)
        remaining_data = pd.concat([remaining_data, remaining])
    
    else:
        # group the data by classes
        class_groups = dataset_ledger.groupby('class_index')

        # dataset_ledger.columns
        for _, group in class_groups:
            # fixed number of random samples per class
            if mode == 1:
                if len(group) >= num_samples:
                    sampled = group.sample(n=num_samples, random_state=SEED)
                else:
                    raise ValueError("insufficient number of samples available for class [{}] of [{}] samples to randomly select [{}] \
                                    samples from. Choose a smaller split".format(group.class_index[0], len(group), num_samples))                    
            
            # % split of the samples per class (randomized)
            elif mode == 2:
                num_samples_class = round(len(group)*pct_split)
                sampled = group.sample(n=num_samples_class, random_state=SEED)
            
            sampled_data.append(sampled)
            remaining = group.drop(sampled.index)
            remaining_data = pd.concat([remaining_data, remaining])

        # concatenate the sampled data
        sampled_data = pd.concat(sampled_data)

    # verify that the splits were correct
    assert len(sampled_data) + len(remaining_data) == total_samples    

    # save the generated test split
    sampled_data.to_csv(test_csv_file, index=False)            

    # save the remaining data as the training split to a new CSV file
    remaining_data.to_csv(train_csv_file, index=False)

if __name__ == "__main__":
    main()