import pathlib

from datasets.preprocess import (
    create_non_label_eval_json,
    get_cm_protocols,
    get_dataset_annotation,
    random_split_train_dev,
)

if __name__ == '__main__':


    # TODO: MAKE THIS PARSE ARGUMENTS
    print('----Start to Process Data -----')



    args = {}
    args['data_type'] = ['labeled', 'unlabeled'][0]
    save_dir = 'processed_data'

    if args['data_type'] == 'labeled':
        print('Start to process labeled data:')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        LA_PRO_DIR = '../data/LA/ASVspoof2019_LA_cm_protocols'
        PRO_FILES = ('ASVspoof2019.LA.cm.train.trn.txt',
                     'ASVspoof2019.LA.cm.dev.trl.txt',
                     'ASVspoof2019.LA.cm.eval.trl.txt')
        DATA_DIR = '../data/'
        FEATURE_NAME = 'cm'
        split_features = get_cm_protocols(
            pro_dir=LA_PRO_DIR,
            pro_files=PRO_FILES,
        )
        get_dataset_annotation(split_features,
                               feature_name=FEATURE_NAME,
                               data_dir=DATA_DIR,
                               save_dir=save_dir,
                               )
        random_split_train_dev(data_dir=save_dir,
                               file=FEATURE_NAME + '_merge.json')
    else:
        print('Start to process unlabeled eval data:')
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        create_non_label_eval_json(
            output_file=str(pathlib.Path(save_dir) / 'cm_eval.json'),
        )


