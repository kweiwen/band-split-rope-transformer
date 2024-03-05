import argparse
import typing as tp
from pathlib import Path

from moisesdb.dataset import MoisesDB
import torch
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
    default= "..//..//dataset//moisesdb",
    help="Path to directory with moisesdb dataset"
)
parser.add_argument(
    '-o',
    '--output-dir',
    type=str,
    required=False,
    default= "D://Project//band-split-rope-transformer//files",
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

def run_program(
        file_path: Path,
        target: str,
        db: MoisesDB,
        sad: SAD,
) -> None:
    """
    Saves track's name and fragments indices to provided .txt file.
    """
    with open(file_path, 'w') as wf:
        for track in tqdm(db):

            if target in list(track.stems.keys()):
                # load audio mixture from track
                y = torch.tensor(track.audio)
                # find indices of salient segments
                indices = sad.calculate_salient_indices(y)
                # write to file
                for line in prepare_save_line(track.id, indices, int(sad.window_size)):
                    wf.write(line)

            else:
                pass
                # TODO:
                #  1. PICK ANY "STEM" INSIDE THE TRACK FOLDER
                #  2. FIND THE LENGTH OF THAT "STEM"
                #  3. CREATE THE EMPTY TENSOR
                #  4. WRITE TO LINE LIKE ABOVE'S TRUE CASE


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
    train_db_subset, test_db_subset = split_dataset(db)


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


if __name__ == '__main__':
    main(
        args.input_dir,
        args.output_dir,
        args.subset,
        args.split,
        args.targets,
        args.sad_cfg_path,
    )

