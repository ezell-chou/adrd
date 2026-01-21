import csv
import random
import argparse
from pathlib import Path
from typing import List, Tuple


class DatasetSampler:
    """
    Image dataset sampler for collecting AI-generated and natural images.
    """

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
    IGNORE_DIRS = {'.idea', '__pycache__', '.git'}
    
    def __init__(
        self,
        img_root: Path,
        save_index: Path,
        n: int = 500,
    ) -> None:
        """
        Initialize the dataset sampler.

        Parameters
        ----------
        img_root : Path
            Root directory path of GenImage dataset.
        save_index : Path
            Output directory for CSV files.
        n : int, optional
            Number of samples per category (ai and nature) for each dataset.
            Default is 500.
        """
        self.genimage_root = Path(img_root)
        self.output_dir = Path(save_index)
        self.samples_per_category = n

        if not self.genimage_root.exists():
            raise ValueError(f"GenImage root directory does not exist: {self.genimage_root}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_name(self, dataset_path: Path) -> str:
        """
        Extract dataset name from directory path.

        Parameters
        ----------
        dataset_path : Path
            Dataset directory path.
            Example: imagenet_ai_0508_adm -> adm

        Returns
        -------
        str
            Extracted dataset name.
        """
        dir_name = dataset_path.name
        parts = dir_name.split('_')
        if len(parts) >= 2:
            return parts[-1]
        return dir_name
    
    def get_image_files(self, category_path: Path) -> List[Path]:
        """
        Retrieve all image files from a category directory.

        Parameters
        ----------
        category_path : Path
            Category directory path (ai or nature).

        Returns
        -------
        List[Path]
            List of image file paths.
        """
        if not category_path.exists():
            return []

        image_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(category_path.glob(f'*{ext}'))
            image_files.extend(category_path.glob(f'*{ext.upper()}'))

        return image_files
    
    def sample_dataset(self, dataset_path: Path) -> Tuple[List[Path], List[Path]]:
        """
        Sample AI-generated and natural images from a single dataset.

        Parameters
        ----------
        dataset_path : Path
            Dataset directory path.

        Returns
        -------
        Tuple[List[Path], List[Path]]
            Tuple of (ai_samples, nature_samples).
        """
        ai_samples = []
        nature_samples = []

        for split_dir in ['train', 'val']:
            split_path = dataset_path / split_dir
            if not split_path.exists():
                continue

            ai_dir = split_path / 'ai'
            if ai_dir.exists():
                ai_files = self.get_image_files(ai_dir)
                ai_samples.extend(ai_files)

            nature_dir = split_path / 'nature'
            if nature_dir.exists():
                nature_files = self.get_image_files(nature_dir)
                nature_samples.extend(nature_files)

        if len(ai_samples) > self.samples_per_category:
            ai_samples = random.sample(ai_samples, self.samples_per_category)

        if len(nature_samples) > self.samples_per_category:
            nature_samples = random.sample(nature_samples, self.samples_per_category)

        return ai_samples, nature_samples
    
    def get_relative_path(self, file_path: Path) -> str:
        """
        Get relative path with respect to the GenImage root directory.

        Parameters
        ----------
        file_path : Path
            Absolute file path.

        Returns
        -------
        str
            Relative path in POSIX format (forward slashes).
        """
        try:
            relative = file_path.relative_to(self.genimage_root)
            return relative.as_posix()
        except ValueError:
            return file_path.name
    
    def determine_is_fake(self, file_path: Path) -> int:
        """
        Determine if an image is AI-generated or natural based on path.

        Parameters
        ----------
        file_path : Path
            File path to analyze.

        Returns
        -------
        int
            1 for AI-generated (fake), 0 for natural (real), -1 for unknown.
        """
        path_str = str(file_path).lower()

        if '/nature/' in path_str or '\\nature\\' in path_str:
            return 0
        elif '/ai/' in path_str or '\\ai\\' in path_str:
            return 1
        else:
            return -1
    
    def save_csv(
        self,
        csv_path: Path,
        samples: List[Path],
        is_fake_value: int,
    ) -> None:
        """
        Save sampled results to CSV file.

        Parameters
        ----------
        csv_path : Path
            Output CSV file path.
        samples : List[Path]
            List of sampled file paths.
        is_fake_value : int
            Value for the IsFake column.
        """
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['num', 'imagepath', 'IsFake'])

            for idx, file_path in enumerate(samples, 1):
                relative_path = self.get_relative_path(file_path)
                writer.writerow([idx, relative_path, is_fake_value])

        print(f"Generated: {csv_path.name} ({len(samples)} records)")
    
    def process_all_datasets(self) -> None:
        """
        Process all datasets in the GenImage root directory.
        """
        dataset_dirs = [
            d for d in self.genimage_root.iterdir()
            if d.is_dir() and d.name not in self.IGNORE_DIRS and d.name.startswith('imagenet_')
        ]

        if not dataset_dirs:
            print(f"Warning: No dataset directories found in {self.genimage_root}")
            return

        dataset_dirs.sort(key=lambda x: x.name)

        print(f"\nFound {len(dataset_dirs)} datasets")
        print("=" * 60)

        total_ai_samples = 0
        total_nature_samples = 0

        for dataset_path in dataset_dirs:
            dataset_name = self.get_dataset_name(dataset_path)
            print(f"\nProcessing dataset: {dataset_path.name} (name: {dataset_name})")
            print("-" * 60)

            ai_samples, nature_samples = self.sample_dataset(dataset_path)

            print(f"  Found AI images: {len(ai_samples)}")
            print(f"  Found natural images: {len(nature_samples)}")

            ai_csv_path = self.output_dir / f"{dataset_name}_ai.csv"
            nature_csv_path = self.output_dir / f"{dataset_name}_nature.csv"

            self.save_csv(ai_csv_path, ai_samples, 1)
            self.save_csv(nature_csv_path, nature_samples, 0)

            total_ai_samples += len(ai_samples)
            total_nature_samples += len(nature_samples)

        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"Total sampled: {total_ai_samples} AI images, {total_nature_samples} natural images")
        print(f"CSV output directory: {self.output_dir}")
        print("=" * 60 + "\n")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sample images from GenImage dataset and generate CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python utils/generate_dataset.py
  
  # Specify custom paths
  python utils/generate_dataset.py --img_root /path/to/images --save_index /path/to/output --n 100
        """
    )
    
    parser.add_argument('--img_root', type=Path, default=Path('D:\\GenImage'),
                       help='Images root directory')
    parser.add_argument('--save_index', type=Path, default=Path('datasets/GenImage'),
                       help='Image index output directory')
    parser.add_argument('--n', type=int, default=20,
                       help='Number of samples per category for each dataset')
    
    return parser.parse_args()


def main():
    args = get_args()

    sampler = DatasetSampler(
        img_root=args.img_root,
        save_index=args.save_index,
        n=args.n
    )
    sampler.process_all_datasets()


if __name__ == '__main__':
    main()
