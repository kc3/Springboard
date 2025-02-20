# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import requests
import zipfile
from shutil import copyfile

@click.command()
@click.argument('input_file_path', type=click.Path(exists=True))
@click.argument('interim_file_path', type=click.Path())
@click.argument('output_file_path', type=click.Path())
def main(input_file_path, interim_file_path, output_file_path):
    """ Runs data processing scripts to turn raw data from (./data/raw) into
        cleaned data ready to be analyzed (saved in ./data/processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('Downloading raw files.')
    # url = 'http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2016/raw/en.zip'
    url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
    file_path = download_file(url, input_file_path, '')
    logging.info('Downloaded file {0}'.format(file_path))

    logger.info('Unzipping file: {0}'.format(file_path))
    unzip_file(file_path, interim_file_path)
    logger.info('Unzipped files: {0}'.format(file_path))

    copyfile('{0}/cornell movie-dialogs corpus/movie_lines.txt'.format(interim_file_path),
             '{0}/movie_lines.txt'.format(interim_file_path))
    copyfile('{0}/cornell movie-dialogs corpus/movie_conversations.txt'.format(interim_file_path),
             '{0}/movie_conversations.txt'.format(interim_file_path))

    # logger.info('making final data set from raw data')


def download_file(url, save_dir, file_prefix):
    """ Downloads file from a given url.

    :param url:
        File to download
    :param save_dir:
        Directory to save the file in
    :param file_prefix:
        File prefix to be added to the name
    :return:
        File path of the downloaded file
    """
    local_filename = url.split('/')[-1]
    file_path = '{0}/{1}{2}'.format(save_dir, file_prefix, local_filename)
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            # filter out keep-alive new chunks
            if chunk:
                f.write(chunk)

    return file_path


def unzip_file(url, save_dir):
    """ Unzips a file at the given url.

    :param url:
        Url to unzip
    :param save_dir:
        Directory to save contents in
    :return:
        None
    """
    with zipfile.ZipFile(url, 'r') as zip_ref:
        zip_ref.extractall(save_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[3]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
