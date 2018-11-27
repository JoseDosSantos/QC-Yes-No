import combine_sources
import clean
import pos_prep
import reclean
import extract_info_from_tag
import config as cfg
from subprocess import run


def main(sets, path='data/'):
    print('Combining raw data sets...')
    combine_sources.main(sets, path)

    print('Cleaning string for Lemmatization and Tagging...')
    clean.main(path)

    print('Splitting the data for Lemmatization and Tagging...')
    pos_prep.main(path)

    print('Creating Lemmas and Pos Tags using RFTagger...')
    run(['java', '-jar', 'C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Classification\\preprocessing\\RFTagger\\rft-java.jar', '-c', 'stts', '-l', 'german', '-x',
         'C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Classification\\preprocessing\\RFTagger\\lib\\german-rft-tagger-lemma-lexicon.txt',
         'C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Classification\\preprocessing\\RFTagger\\lib\\german.par', path + 'all_pre_pos.txt', path + 'pos_tagged_raw.txt'], shell=True)

    print('Further streamlining data...')
    reclean.main(path)

    print('Extracting final data set...')
    extract_info_from_tag.main(path)

    print('Process finished.')

if __name__ == '__main__':
    sets = {'amazon': cfg.AMAZON,
    'twitter': cfg.TWEETS,
    'all': ['data\\twitter.csv', 'data\\amazon.csv']}
    main(sets)