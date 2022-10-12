import logging
import json
import alm
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Word Analogy Test')
parser.add_argument('--ckpt-dir', type=str)
parser.add_argument('--ckpt-tag', type=str)
parser.add_argument('--model-tag', type=str)
parser.add_argument('--gpu-id', type=int)
parser.add_argument('--cache-dir', type=str, default='YOUR/CASH/DIRPATH')
parser.add_argument('--export-dir', type=str, default='./experiments_results_ppl_sph')

args = parser.parse_args()

export_prefix = 'experiment.ppl_variants.{}'.format(args.model_tag)

SKIP_INFERENCE = False  # skip inference step
SKIP_GRID_SEARCH = False  # skip grid search
SKIP_MERGE = False  # skip merging result
SKIP_EXPORT_PREDICTION = False  # skip export prediction

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('')
alm.util.fix_seed(1234)
all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
data = ['sat', 'u2', 'u4', 'google', 'bats']
#models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 256), ('bert-large-cased', 32, 1024)]
"""
MODEL = {
	'bert': [('bert-base-uncased', 32, 512)],
	'sphbert': [('sphbert-base-uncased', 32, 512)],
	'roberta': [('roberta-base', 32, 512)],
	'sphroberta': [('sphroberta-base', 32, 512)],
	'bert_empty': [('bert-base-uncased', 32, 512)],
	'sphbert_empty': [('sphbert-base-uncased', 32, 512)],
	'roberta_empty': [('roberta-base', 32, 512)],
	'sphroberta_empty': [('sphroberta-base', 32, 512)],
	'bert_l': [('bert-large-uncased', 32, 512)],
	'sphbert_l': [('sphbert-large-uncased', 32, 512)],
	'roberta_l': [('roberta-large', 32, 512)],
	'sphroberta_l': [('sphroberta-large', 32, 512)],
	'bert_empty_l': [('bert-large-uncased', 32, 512)],
	'sphbert_empty_l': [('sphbert-large-uncased', 32, 512)],
	'roberta_empty_l': [('roberta-large', 32, 512)],
	'sphroberta_empty_l': [('sphroberta-large', 32, 512)],
}
"""
MODEL = {
	'bert': [('bert-large-cased', 32, 1024)],
	'roberta': [('roberta-large', 32, 512)],
	'bert_wiki': [('bert-large-cased', 32, 1024)],
	'roberta_wiki': [('roberta-large', 32, 512)],
	'sphbert': [('sphbert-large-cased', 32, 1024)],
	'sphroberta': [('sphroberta-large', 32, 512)],
}

models = MODEL[args.model_tag]
scoring_method = ['ppl_hypothesis_bias', 'ppl_marginal_bias', 'ppl_based_pmi']
#export_prefix = 'experiment.ppl_variants'
ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_tag)

if not SKIP_INFERENCE:
    logging.info('###############################################################')
    logging.info('# Run LM inference to get logit (both of valid and test sets) #')
    logging.info('###############################################################')
    no_inference = False
    for _model, _max_length, _batch in models:
        scorer = alm.RelationScorer(model=_model, max_length=_max_length, ckpt_path=ckpt_path,
									cache_dir=args.cache_dir, gpu_id=args.gpu_id)
        for _data in data:
            for _temp in all_templates:
                for test in [True, False]:
                    for score in scoring_method:
                        if 'gpt' in _model and score == 'ppl_hypothesis_bias':
                            continue
                        scorer.analogy_test(
                            scoring_method=score,
                            data=_data,
                            template_type=_temp,
                            batch_size=_batch,
                            no_inference=no_inference,
                            negative_permutation=True,
                            skip_scoring_prediction=True,
                            test=test,
							export_prefix=export_prefix,
							export_dir=args.export_dir
                        )
                        scorer.release_cache()

if not SKIP_GRID_SEARCH:
    logging.info('######################################################################')
    logging.info('# Get prediction on each configuration (both of valid and test sets) #')
    logging.info('######################################################################')
    positive_permutation_aggregation = [
        'max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7'
    ]
    negative_permutation_aggregation = [
        'max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7',
        'index_8', 'index_9', 'index_10', 'index_11'
    ]
    negative_permutation_weight = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    weight_head = [-0.4, -0.2, 0, 0.2, 0.4]
    weight_tail = [-0.4, -0.2, 0, 0.2, 0.4]
    ppl_based_pmi_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1']
    ppl_based_pmi_alpha = [-0.4, -0.2, 0, 0.2, 0.4]
    no_inference = True

    for _model, _max_length, _batch in models:
        scorer = alm.RelationScorer(model=_model, max_length=_max_length, ckpt_path=ckpt_path,
									cache_dir=args.cache_dir, gpu_id=args.gpu_id)
        for _data in data:
            for _temp in all_templates:
                for test in [False, True]:
                    for score in scoring_method:
                        if 'gpt' in _model and score == 'ppl_hypothesis_bias':
                            continue
                        scorer.analogy_test(
                            no_inference=no_inference,
                            scoring_method=score,
                            data=_data,
                            template_type=_temp,
                            batch_size=_batch,
                            export_prefix=export_prefix,
                            ppl_hyp_weight_head=weight_head,
                            ppl_hyp_weight_tail=weight_tail,
                            ppl_mar_weight_head=weight_head,
                            ppl_mar_weight_tail=weight_tail,
                            ppl_based_pmi_aggregation=ppl_based_pmi_aggregation,
                            ppl_based_pmi_alpha=ppl_based_pmi_alpha,
                            negative_permutation=True,
                            positive_permutation_aggregation=positive_permutation_aggregation,
                            negative_permutation_aggregation=negative_permutation_aggregation,
                            negative_permutation_weight=negative_permutation_weight,
                            test=test,
							export_dir=args.export_dir)
                        scorer.release_cache()

    alm.export_report(export_prefix=export_prefix, export_dir=args.export_dir)
    alm.export_report(export_prefix=export_prefix, export_dir=args.export_dir, test=True)

if not SKIP_MERGE:
    logging.info('####################################')
    logging.info('# Merge validation and test result #')
    logging.info('####################################')
    df_val = alm.get_report(export_prefix=export_prefix, export_dir=args.export_dir)
    df_val = df_val.sort_values(by=list(df_val.columns))

    df_test = alm.get_report(export_prefix=export_prefix, export_dir=args.export_dir, test=True)
    df_test = df_test.sort_values(by=list(df_val.columns))

    accuracy_val = df_val.pop('accuracy').to_numpy()
    accuracy_test = df_test.pop('accuracy').to_numpy()
    assert df_val.shape == df_test.shape

    df_test['accuracy_validation'] = accuracy_val
    df_test['accuracy_test'] = accuracy_test

    summary = {}

    for d in data:
        df_test_ = df_test[df_test.data == d]
        val, test = alm.get_dataset_raw(d)
        df_test_['accuracy'] = (df_test_['accuracy_validation'] * len(val) + df_test_['accuracy_test'] * len(test)) / (len(val) + len(test))
        df_test_ = df_test_.sort_values(by=['accuracy'], ascending=False)
        df_test_.to_csv('{}/summary/{}.full.{}.csv'.format(args.export_dir, export_prefix, d))
        logging.info('Top 3 in {}'.format(d))
        logging.info('\n{}'.format(df_test_['accuracy'].head(3)))
        summary[d] = {}
        for m, _, _ in models:
            df_test__ = df_test_[df_test_['model'] == m]
            acc_full = float(df_test__.sort_values(by=['accuracy'], ascending=False)['accuracy'].head(1))
            acc_val = float(df_test__.sort_values(by=['accuracy_validation'], ascending=False)['accuracy'].head(1))
            summary[d][m] = {'full': acc_full, 'validation': acc_val}
    with open('{}/summary/{}.top.json'.format(args.export_dir, export_prefix), 'w') as f:
        json.dump(summary, f)

if not SKIP_EXPORT_PREDICTION:
    logging.info('###############################################')
    logging.info('# Export predictions for qualitative analysis #')
    logging.info('###############################################')
    # get prediction of what achieves the best validation accuracy
    methods = ['ppl_marginal_bias', 'ppl_hypothesis_bias', 'ppl_based_pmi']
    for d in data:
        logging.info('DATASET: {}'.format(d))
        df_test_full = pd.read_csv('{}/summary/{}.full.{}.csv'.format(args.export_dir, export_prefix, d),
                                   index_col=0)
        for method in methods:
            df_test = df_test_full[df_test_full.scoring_method == method]
            for _model, _max_length, _batch in models:
                df_tmp = df_test[df_test.model == _model]
                df_tmp = df_tmp.sort_values(by=['accuracy_validation'], ascending=False)
                if len(df_tmp) == 0:
                    continue
                acc_val = list(df_tmp.head(1)['accuracy_validation'])[0]
                acc = df_tmp[df_tmp.accuracy_validation == acc_val].sort_values(by=['accuracy_test'])
                acc_test = list(acc['accuracy_test'])
                acc_test = acc_test[int(len(acc_test) / 2)]
                best_configs = df_tmp[df_tmp.accuracy_test == acc_test]
                config = json.loads(best_configs.iloc[0].to_json())
                logging.info("* {}/{}".format(method, _model))
                logging.info("\t * accuracy (valid): {}".format(config.pop('accuracy_validation')))
                logging.info("\t * accuracy (test) : {}".format(config.pop('accuracy_test')))
                logging.info("\t * accuracy (full) : {}".format(config.pop('accuracy')))
                scorer = alm.RelationScorer(model=config.pop('model'), max_length=config.pop('max_length'),
											cache_dir=args.cache_dir, gpu_id=args.gpu_id, ckpt_path=ckpt_path)
                scorer.analogy_test(test=True, export_prediction=True, no_inference=True, export_prefix=export_prefix,
									export_dir=args.export_dir,
                                    **config)
                scorer.release_cache()

