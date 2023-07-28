import os
import requests
import argparse
import logging
import mimetypes
import csv
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser(description="Download images from the web.")
    parser.add_argument("--output_folder", type=str,
                        default='../laion', help="Folder for the downloaded images")
    parser.add_argument("--path", type=str,
                        default='./laion_train_real.csv', help="Path of the csv file")
    parser.add_argument("--images_per_folder", type=int,
                        default=1000,
                        help="Number of images per folder")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()
    return args

def download_image(url, filename):
    try:
        response = requests.get(url, timeout=2)
        content_type = response.headers['content-type']
        extension = mimetypes.guess_extension(content_type)
        # check if filename + extension is already present
        if os.path.exists(filename + extension):
            return True
        with open(filename + extension, 'wb') as fo:
            fo.write(response.content)
        return True
    except:
        logging.info(f'Error downloading {url}')
        return False


def download_laion_images(args, download_folder):
    counter_folder = 0
    with open(args.path, 'r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        next(csv_reader) # skip header
        for index, row in enumerate(csv_reader):
            sample_id = float(row[0])
            url = row[1]
            if index % 100000 == 0:
                logging.info(f'Downloaded {index} of 1000000 images')
            logging.debug(f'Downloading {sample_id} ...')

            if index % args.images_per_folder == 0:
                new_id = str(counter_folder).zfill(6)
                current_folder = os.path.join(download_folder, f"part-{new_id}")
                if not os.path.exists(current_folder):
                    os.makedirs(current_folder)
                    logging.debug(f"Created folder: {current_folder}")
                counter_folder += 1

            download_image(url, os.path.join(current_folder, f"{int(sample_id)}"))


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, filename=f'{args.output_folder}/download.log', filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    print("All logs are saved at " + f'{args.output_folder}/download.log')
    download_folder = os.path.join(args.output_folder, 'real-images')
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    download_laion_images(args, download_folder)
    logging.info(f'Downloaded all images')