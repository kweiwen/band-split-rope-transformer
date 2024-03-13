import argparse
import typing as tp
from pathlib import Path

from moisesdb.dataset import MoisesDB
import torch
import torchaudio
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from data import SAD



parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input-dir',
    type=str,
    required=False,
    default= "../../dataset/moisesdb",
    help="Path to directory with moisesdb dataset"
)
parser.add_argument(
    '-o',
    '--output-dir',
    type=str,
    required=False,
    default= "./files",
    help="Path to directory where output .txt file is saved"
)
parser.add_argument(
    '--subset',
    type=str,
    required=False,
    default='test',
    help="Train/test subset of dataset to process"
)
parser.add_argument(
    '--split',
    type=str,
    required=False,
    default='train',
    help="Train/valid split of train dataset. Used if subset=train"
)
parser.add_argument(
    '--sad-cfg-path',
    type=str,
    required=False,
    default="./conf/sad/default.yaml",
    help="Path to Source Activity Detection config file"
)
parser.add_argument(
    '-t',
    '--targets',
    nargs='+',
    required=False,
    default=["vocals"],
    help="Target source. SAD will save salient fragments of vocal audio."
)
args = parser.parse_args()


def prepare_save_line(
        track_name: str,
        start_indices: torch.Tensor,
        window_size: int
) -> tp.Iterable[str]:
    """
    Creates string in format TRACK_NAME START_INDEX END_INDEX.
    """
    for i in start_indices:
        save_line = f"{track_name}\t{i}\t{i + window_size}\n"
        yield save_line

def split_dataset(db: MoisesDB, ratio: float = 0.2):
    # get total length of datasets
    n_songs = len(db)

    # size of test set
    n_test_songs = int(n_songs * ratio)

    # generate shuffled indices
    indices = np.arange(n_songs)
    np.random.shuffle(indices)

    # assign shuffled indices to dataset
    test_set = [db[i] for i in indices[:n_test_songs]]
    train_set = [db[i] for i in indices[n_test_songs:]]

    return test_set, train_set

def pad_tracks_to_equal_length(db_dir:str = 'D:\zengkuiwen\Desktop\REFACTOR'):
    # each subdirectory may contain one or more WAV files,
    # each representing a specific recording or sample relevant to the instrument category.
    #
    # ├─02ee37da-eea3-42b4-83bf-ab7f243afa13
    # │  │  data.json
    # │  │
    # │  ├─bass
    # │  │      6ffbc14a-572b-43d7-bbbb-1b5e7fc62889.wav
    # │  │
    # │  ├─bowed_strings
    # │  │      df7686dc-c383-4727-bf1d-1bf1e09dec18.wav
    # │  │
    # │  ├─drums
    # │  │      01eb58f6-a186-4d5e-adb3-f47189f4a5e9.wav
    # │  │      18d5c271-0d2a-4d94-b069-6cb6b8e80e80.wav
    # │  │      3c8c6f15-9418-4352-acfa-53b29a494dad.wav
    # │  │      91f922b7-3fae-4f87-9da7-837c5537b5dc.wav
    # │  │      c2f6171e-c52f-49fd-8d7b-ff00dad82820.wav
    # │  │
    # │  ├─guitar
    # │  │      43896fa4-9f20-4603-9254-074347e76631.wav
    # │  │      c08b4e56-1010-4f3e-85f4-dcfc2bde216f.wav
    # │  │
    # │  ├─percussion
    # │  │      55b0cc6d-8e71-4291-8c6e-d3fffc1e06a3.wav
    # │  │
    # │  ├─piano
    # │  │      5df8b123-d37b-4ff7-8abe-721084b9ea12.wav
    # │  │      a1f763a8-4ee4-496d-a2fa-3f94fe605755.wav
    # │  │
    # │  └─vocals
    # │          7c555f5b-2ef6-44af-acc3-9484d12412b0.wav
    # next track...

    db = MoisesDB(
        data_path=db_dir,
        sample_rate=44100
    )

    for track in db:
        for stem_key, stem in track.stems.items():

            # detect the length between stem and mixture
            if not len(stem[0]) == len(track.audio[0]):
                # there are multiple sources in a single stem
                for source_key, element_path in track.sources[stem_key].items():
                    temp, sample_rate = torchaudio.load(element_path[0])
                    # detect the length between source and mixture
                    if not (len(temp[0]) == len(track.audio[0])):
                        # calculate difference
                        padding_needed = len(track.audio[0]) - len(temp[0])
                        # create padding
                        padding = torch.zeros((temp.shape[0], padding_needed))
                        # concatenate padding
                        temp_padded = torch.cat((temp, padding), 1)
                        # save file
                        torchaudio.save(element_path[0], temp_padded, 44100)

def run_program(
        savepath: Path,
        target: str,
        db: list,
        sad: SAD,
) -> None:
    """
    Saves track's name and fragments indices to provided .txt file.
    """
    with open(savepath, 'w') as wf:
        for track in tqdm(db):
            filepath_template = str(Path(track.path) / "{}.wav")
            filepath = filepath_template.format("vocals")
            # get audio data and transform to torch.Tensor
            y, sr = torchaudio.load(filepath)
            # find indices of salient segments
            indices = sad.calculate_salient_indices(y)
            # write to file
            for line in prepare_save_line(track.id, indices, sad.window_size):
                wf.write(line)
    return None

def main(
        db_dir: str,
        save_dir: str,
        subset: str,
        split: tp.Optional[str],
        targets: tp.List[str],
        sad_cfg_path: DictConfig
) -> None:

    # initialize moisesdb parser
    db = MoisesDB(
        data_path=db_dir,
        sample_rate=44100
    )
    test_db_subset, train_db_subset = split_dataset(db)

    # initialize Source Activity Detector
    sad_cfg = OmegaConf.load(sad_cfg_path)
    sad = SAD(**sad_cfg)

    # initialize directories where to save indices
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for target in targets:
        # file path
        train_file_path = save_dir / f"{target}_moisesdb_train.txt"
        test_file_path = save_dir / f"{target}_moisesdb_test.txt"

        # segment data and save indices to .txt file
        run_program(train_file_path, target, train_db_subset, sad)
        run_program(test_file_path, target, test_db_subset, sad)

    return None

def check_waveform_consistency(db_dir: str, target: str, index: int):
    '''
    '''
    # initialize moisesdb parser
    db = MoisesDB(
        data_path=db_dir,
        sample_rate=44100
    )

    temp = db[index].stems[target]
    waveforms = []
    for source_key, element_path in db[index].sources[target].items():
        data, sr = torchaudio.load(element_path[0])
        waveforms.append(data)

    tensor = sum(waveforms)
    tensor_to_array = tensor.numpy()
    are_equal = np.array_equal(temp, tensor_to_array)
    are_close = np.allclose(temp.astype(float), tensor_to_array.astype(float))

    return are_equal, are_close

if __name__ == '__main__':
    main(
        args.input_dir,
        args.output_dir,
        args.subset,
        args.split,
        args.targets,
        args.sad_cfg_path,
    )

