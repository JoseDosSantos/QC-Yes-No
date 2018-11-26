import combine_sources
import clean
import pos_prep
import reclean
import extract_info_from_tag
import config as cfg
from subprocess import run


def main():
        sets = {'amazon': cfg.AMAZON,
                'twitter': cfg.TWEETS,
                'all': ['data\\twitter.csv', 'data\\amazon.csv']}

        print('Combining raw data sets...')
        combine_sources.main(sets)

        print('Cleaning string for Lemmatization and Tagging...')
        clean.main()

        print('Splitting the data for Lemmatization and Tagging...')
        pos_prep.main()

        print('Creating Lemmas and Pos Tags using RFTagger...')
        run(['java', '-jar', 'RFTagger\\rft-java.jar', '-c', 'stts', '-l', 'german', '-x', 'RFTagger\\lib\\german-rft-tagger-lemma-lexicon.txt', 'RFTagger\\lib\\german.par', 'data\\all_pre_pos.txt', 'data\\pos_tagged_raw.txt'], shell=True)

        print('Further streamlining data...')
        reclean.main()

        print('Extracting final data set...')
        extract_info_from_tag.main()

        print('Process finished.')

if __name__ == '__main__':
    main()