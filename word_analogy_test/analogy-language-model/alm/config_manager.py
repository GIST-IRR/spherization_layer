""" configuration manager for `scoring_function.RelationScorer` """
import os
import random
import json
import string
import logging
import pickle
from glob import glob
from typing import List


def get_random_string(length: int = 6, exclude: List = None):
    tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    if exclude:
        while tmp in exclude:
            tmp = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    return tmp


def safe_open(_file):
    with open(_file, 'r') as f:
        return json.load(f)


class ConfigManager:
    """ configuration manager for `scoring_function.RelationScorer` """

    def __init__(self, export_dir: str, test: bool, **kwargs):
        """ configuration manager for `scoring_function.RelationScorer` """
        self.config = kwargs
        self.prefix = 'test' if test else 'valid'
        logging.info(' * configuration\n' +
                     '\n'.join(list(map(lambda x: '{} : {}'.format(x[0], x[1]), self.config.items()))))
        cache_dir = os.path.join(export_dir, kwargs['data'], kwargs['model'], kwargs['scoring_method'])
        self.pmi_logits = {'positive': None, 'negative': None}
        self.flatten_score = {'positive': None, 'negative': None}
        self.flatten_score_mar = {'positive': None, 'negative': None}

        ex_configs = {i: safe_open(i) for i in glob('{}/*/config.json'.format(cache_dir))}
        same_config = list(filter(lambda x: x[1] == self.config, ex_configs.items()))
        if len(same_config) != 0:
            assert len(same_config) == 1, 'duplicated config found {}'.format(same_config)
            self.cache_dir = same_config[0][0].replace('config.json', '')

            # load intermediate score
            for i in ['positive', 'negative']:
                _file = os.path.join(self.cache_dir, 'flatten_score.{}.{}.pkl'.format(i, self.prefix))
                if os.path.exists(_file):
                    with open(_file, "rb") as fp:  # Unpickling
                        self.flatten_score[i] = pickle.load(fp)
                    logging.debug('load flatten_score.{} from {}'.format(i, _file))

                # load stats for ppl_pmi
                _file = os.path.join(self.cache_dir, 'flatten_score_mar.{}.{}.pkl'.format(i, self.prefix))
                if os.path.exists(_file):
                    with open(_file, "rb") as fp:  # Unpickling
                        self.flatten_score_mar[i] = pickle.load(fp)
                    logging.debug('load flatten_score_mar.{} from {}'.format(i, _file))

            # load intermediate score for PMI specific
            if self.config['scoring_method'] in ['pmi']:
                for i in ['positive', 'negative']:
                    # skip if full score is loaded
                    if i in self.flatten_score.keys():
                        continue
                    self.pmi_logits[i] = {}
                    for _file in glob(os.path.join(self.cache_dir, 'pmi.{}.*.{}pkl'.format(i, self.prefix))):
                        if os.path.exists(_file):
                            k = _file.split('pmi.{}.'.format(i))[-1].replace('.pkl', '')
                            with open(_file, "rb") as fp:  # Unpickling
                                self.pmi_logits[i][k] = pickle.load(fp)
                            logging.debug('load pmi.{} from {}'.format(i, _file))

        else:
            self.cache_dir = os.path.join(cache_dir, get_random_string())

    def __cache_init(self):
        assert self.cache_dir is not None
        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists('{}/config.json'.format(self.cache_dir)):
            with open('{}/config.json'.format(self.cache_dir), 'w') as f:
                json.dump(self.config, f)

    def cache_scores_pmi(self, logit_name: str, pmi_logit: List, positive: bool = True):
        self.__cache_init()
        prefix = 'positive' if positive else 'negative'
        with open('{}/pmi.{}.{}.{}.pkl'.format(self.cache_dir, prefix, logit_name, self.prefix), "wb") as fp:
            pickle.dump(pmi_logit, fp)

    def cache_scores(self, flatten_score: List, positive: bool = True):
        """ cache scores """
        self.__cache_init()
        prefix = 'positive' if positive else 'negative'
        with open('{}/flatten_score.{}.{}.pkl'.format(self.cache_dir, prefix, self.prefix), "wb") as fp:
            pickle.dump(flatten_score, fp)
