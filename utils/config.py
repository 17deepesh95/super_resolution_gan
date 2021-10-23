IMAGE_OUTPUT_PATH = 'images'
MODEL_SAVE_DIR = 'models'
LOG_FILE_OUTPUT = 'log.txt'
ROOT_DATA_DIR = r"D:\projects\SRGAN\data"

CONFIG = {
    'epoch': 0,  # Change if want to start for a particular number
    'n_epochs': 200,  # Number of epochs to be trained on
    'dataset_name': 'img_align_celeba',  # Name of the folder that contains the dataset
    'batch_size': 10,  # Batch size,
    'lr': 0.0002,  # Learning Rate
    'b1': 0.5, 'b2': 0.999, 'decay_epoch': 100,
    'n_cpu': 0,  # Number of parallel process (+1)
    'hr_height': 256,  # output height of super resolution image
    'hr_width': 256,  # output width of super resolution image
    'channels': 3,  # output channels of super resolution image
    'sample_interval': 1000,  # check output after n steps
    'checkpoint_interval': 10  # save model after n epochs
}
