import glob
import os
import random
from pathlib import Path
import json
from speechbrain.dataio.dataio import read_audio

SAMPLERATE = 16000



def get_cm_protocols(pro_dir = '../data/LA/ASVspoof2019_LA_cm_protocols',
                     pro_files = ('ASVspoof2019.LA.cm.train.trn.txt',
                                 'ASVspoof2019.LA.cm.dev.trl.txt',
                                 'ASVspoof2019.LA.cm.eval.trl.txt')
                     ):
    split_protocols = []

    for file in pro_files:
        cm_features = {}
        protocol_path = os.path.join(pro_dir, file)
        with open(protocol_path, 'r') as f:
            cm_pros = f.readlines()
        for line_number, pro in enumerate(cm_pros, start=1):
            pro = pro.strip().split()
            if len(pro) < 5:
                raise ValueError(
                    "Invalid protocol line {} in {}: {}".format(
                        line_number, protocol_path, pro
                    )
                )
            speaker_id = pro[0]
            auto_file_name = pro[1]
            system_id = pro[3]
            key = pro[4]

            cm_features[auto_file_name] = {
                # 'id':auto_file_name,
                'speaker_id': speaker_id,
                'system_id': system_id,
                'key': key
            }
        split_protocols.append(cm_features)
    # [train, dev, eval]
    if len(split_protocols) != 3:
        raise ValueError("Expected train, dev, and eval protocol files.")
    return {
        'train':split_protocols[0],
        'dev': split_protocols[1],
        'eval': split_protocols[2],
    }

def get_dataset_annotation(split_features,
                           feature_name = 'cm',
                           data_dir = '../data',
                           file_name = 'ASVspoof2019',
                           eval_file_name = None,
                           data_type = 'LA',
                           voice_dir = 'flac',
                           save_dir = 'processed_data/',
                           ):
    os.makedirs(save_dir, exist_ok=True)
    if eval_file_name is None:
        eval_file_name = file_name
    for split in ['train','dev', 'eval']:
        current_file_name = eval_file_name if split == 'eval' else file_name
        features = split_features[split]
        # features have unique id which map to file id
        subfolder = '{type}/{file}_{type}_{split}'.format(file=current_file_name,
                                                   type=data_type,
                                                   split=split)
        path = os.path.join(os.path.join(os.path.join(data_dir, subfolder), voice_dir), '*.flac')
        flac_files = glob.glob(path,  recursive=True)
        create_json(split, flac_files, features, save_dir, feature_name)

    # merge train and dev
    with open(os.path.join(save_dir, feature_name + '_train.json'), 'r') as f:
        train_data = json.load(f)
    with open(os.path.join(save_dir, feature_name + '_dev.json'), 'r') as f:
        dev_data = json.load(f)
    print('train set has %d data'%(len(train_data.keys())))
    print('dev set has %d data' % (len(dev_data.keys())))
    merged_data  = {**train_data, **dev_data}
    print('merged data has %d data' % (len(merged_data.keys())))
    with open(os.path.join(save_dir, feature_name + '_merge.json'), 'w') as f:
        json.dump(merged_data, f)

def create_json(split, files, features, save_dir, feature_name):
    annotations = {}
    n_miss = 0
    for file_path in files:
        signal = read_audio(file_path)
        duration = signal.shape[0] / SAMPLERATE

        id = Path(file_path).stem
        #TODO: SOME id not is features
        if id in features:
            annotations[id] = {
                'file_path': file_path,
                'duration': duration
            }
            for k in features[id]:
                annotations[id][k] = features[id][k]
        else:
            n_miss += 1
    print('%d files missed description in protocol file in %s set'%(n_miss, split))
    output_file = os.path.join(save_dir, feature_name + '_' + split + '.json')
    with open(output_file, 'w') as f:
        json.dump(annotations, f)
        print('Features are saved to %s'%(output_file))

def random_split_train_dev(data_dir = 'processed_data',
                           file = 'cm_merge.json',
                           split_ration = (0.9,0.1),
                           seed = 1243
                           ):
    random.seed(seed)
    if abs(sum(split_ration) - 1.0) > 1e-8:
        raise ValueError('split_ration must sum to 1.0')
    with open(os.path.join(data_dir, file), 'r') as f:
        data = json.load(f)
    # TODO: MAYBE WE CAN HAVE A MORE SMART WAY TO SPLIT DATA
    ids = list(data.keys())
    random.shuffle(ids)
    train_data, dev_data = {}, {}
    print(len(ids))
    print(split_ration[0])
    break_point = int(len(ids)*split_ration[0])
    print(break_point)
    for i in ids[ : break_point]:
        train_data[i] = data[i]
    for i in ids[break_point:]:
        dev_data[i] = data[i]
    assert len(train_data)+len(dev_data) == len(data)
    with open(os.path.join(data_dir, 'new_train.json'),'w') as f:
        json.dump(train_data, f)
    with open(os.path.join(data_dir, 'new_dev.json'),'w') as f:
        json.dump(dev_data, f)
    print('new train/dev set are saved to %s'%(data_dir))

def create_non_label_eval_json(pro_file = '../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt',
                               data_dir = '../data/LA/ASVspoof2021_LA_eval/flac/',
                               output_file = './processed_data/cm_eval.json'
                               ):


    with open(pro_file, 'r') as f:
        data = f.readlines()

    j = {}
    for p in data:
        p = p.replace('\n', '')
        j[p] = {}
        j[p]['file_path'] = os.path.join(data_dir, p + '.flac')
        # Just for format
        j[p]['key'] = 'spoof'

    for k in j:
        signal = read_audio(j[k]['file_path'])
        duration = signal.shape[0] / 16000
        j[k]['duration'] = duration

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(j, f)
