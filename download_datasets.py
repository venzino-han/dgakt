import gdown
import argparse


assist_files_dict ={
    'assist/assist.csv' : '1YWuiE2wYhepN7P6Jo51mproW7_sEANrO',
    'assist/questions.csv' : '1JaZDZC0JmOqdS5g1WJt24_sZDt_art-N',
    'assist/test_df.csv' : '12gyfuF_hnXmBIRMmJ5bl2DZhHBKnG9gC',
    'assist/train_df.csv' : '1U1lKqrZnaqmFXD8mvWs_QcFBQGqpH-5r',
}

ednet_file_dict = {
    'ednet/train_30m.csv' : '1XlTBPBFYEzzy4dUhAYFC78mKGnXmz9X_',
    'ednet/questions.csv' : '1drZV1NkJDuufGkIUDJdeeyW7O61XnWkm',
    'ednet/test_df.csv' : '1QlxPWagiXMYTLdKlZcbyxl8hCideLMWC',
    'ednet/train_df.csv' : '1oVzr0mRAb1hnwbmCFZQdE1T-RUSEGO0L',
}

junyi_file_dict = {
    'junyi/junyi.csv' : '1sGs__0483yE8P_JG8f_eutXSqzGq-MLp',
    'junyi/questions.csv' : '1v7FVMvjH1RcuUfzRS0ND-iNeUxUCVEWW',
    'junyi/test_df.csv' : '1nR63gkQ0FqYwW7gqSmq43zJa4V4-hnDn',
    'junyi/train_df.csv' : '18DWRb_b5KECU1Vtqy-WfpYxr0TmOV3AT',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, default="all")
    parser.add_argument("-p","--path", type=str, default="./data")
    args = parser.parse_args()
    if args.dataset == 'all':
        dataset_files_dict = assist_files_dict | ednet_file_dict 
        dataset_files_dict = dataset_files_dict | junyi_file_dict
    if args.dataset == 'assist':
        dataset_files_dict = assist_files_dict
    for output, url  in dataset_files_dict.items():
        gdown.download(f'https://drive.google.com/uc?id={url}', f'{output}', quiet=False)
