import os
import logging
import json
from itertools import product
import alm
import argparse

parser = argparse.ArgumentParser(description='Word Analogy Test')
parser.add_argument('--ckpt-dir', type=str)
parser.add_argument('--ckpt-tag', type=str)
parser.add_argument('--model-tag', type=str)
parser.add_argument('--gpu-id', type=int)
parser.add_argument('--cache-dir', default='YOUR/CACHE/DIRPATH', type=str)
parser.add_argument('--export-dir', type=str, default='./experiments_results')

args = parser.parse_args()

export_prefix = 'experiment.scoring_comparison.{}'.format(args.model_tag)

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
methods_mlm = ['pmi_feldman', 'ppl_head_masked', 'ppl_tail_masked', 'ppl_add_masked', 'ppl_hypothesis_bias',
			   'embedding_similarity']
all_templates = ['is-to-what', 'is-to-as', 'rel-same', 'what-is-to', 'she-to-as', 'as-what-same']
#all_templates = ['is-to-as']

data = ['sat', 'u2', 'u4', 'google', 'bats']
#data = ['google']
#models = [('roberta-large', 32, 512), ('gpt2-xl', 32, 128), ('bert-large-cased', 32, 1024)]
#models = [('roberta-large', 32, 512)]
#models = [('bert-base-uncased', 32, 512), ('sphbert-base-uncased', 32, 512)]
#models = [('sphbert-base-uncased', 32, 512)]
#models = [('bert-base-uncased', 32, 512)]

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

models = MODEL[args.model_tag]
ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_tag)


""" BACKUP
SKIP_INFERENCE = False
SKIP_PPL = False
SKIP_GRID_SEARCH = False
SKIP_TEST = False
SKIP_DEFAULT = True
"""
SKIP_INFERENCE = False
SKIP_PPL = False
SKIP_GRID_SEARCH = False
SKIP_TEST = False
SKIP_DEFAULT = True

if not SKIP_INFERENCE:
	logging.info('################################################')
	logging.info('# Run LM inference to get logit (on valid set) #')
	logging.info('################################################')
#	methods = ['pmi_feldman', 'embedding_similarity', 'ppl', 'ppl_based_pmi', 'ppl_head_masked', 'ppl_tail_masked']
	methods = ['embedding_similarity', 'ppl']
	no_inference = False
	for _model, _max_length, _batch in models:
		for scoring_method in methods:
			if 'gpt' in _model and scoring_method in methods_mlm:
				continue
			scorer = alm.RelationScorer(model=_model, max_length=_max_length, ckpt_path=ckpt_path, gpu_id=args.gpu_id, cache_dir=args.cache_dir)
			for _data in data:
				for _temp in all_templates:
					scorer.analogy_test(
						scoring_method=scoring_method,
						data=_data,
						template_type=_temp,
						batch_size=_batch,
						no_inference=no_inference,
						skip_scoring_prediction=True,
						export_dir=args.export_dir)
					scorer.release_cache()

if not SKIP_PPL:
	logging.info('##########################################')
	logging.info('# Get perplexity baseline (on valid set) #')
	logging.info('##########################################')
#	export_prefix = 'experiment.scoring_comparison.ppl_baseline'
	no_inference = False
	for _model, _max_length, _batch in models:
		scorer = alm.RelationScorer(model=_model, max_length=_max_length, ckpt_path=ckpt_path, gpu_id=args.gpu_id, cache_dir=args.cache_dir)
		for d in data:
			scorer.analogy_test(data=d, template_type='is-to-as', scoring_method='ppl',
								batch_size=_batch,
								export_dir=args.export_dir,
								export_prefix=export_prefix,
								no_inference=no_inference)
			scorer.analogy_test(data=d, template_type='is-to-as', scoring_method='ppl',
								batch_size=_batch,
								export_dir=args.export_dir,
								export_prefix=export_prefix,
								no_inference=no_inference, test=True)

			scorer.release_cache()
	alm.export_report(export_prefix=export_prefix, export_dir=args.export_dir)
	alm.export_report(export_prefix=export_prefix, export_dir=args.export_dir, test=True)

if not SKIP_GRID_SEARCH:
	logging.info('#######################################################')
	logging.info('# Get prediction on each configuration (on valid set) #')
	logging.info('#######################################################')
#	methods = ['pmi_feldman', 'embedding_similarity', 'ppl', 'ppl_based_pmi', 'ppl_marginal_bias',
#			   'ppl_hypothesis_bias']
#	methods += ['ppl_head_masked', 'ppl_tail_masked', 'ppl_add_masked']
	methods = ['embedding_similarity', 'ppl']
	positive_permutation_aggregation = [
		'max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7'
	]
	pmi_feldman_aggregation = [
		'max', 'mean', 'min', 'index_0', 'index_1', 'index_2', 'index_3', 'index_4', 'index_5', 'index_6', 'index_7',
		'index_8', 'index_9', 'index_10', 'index_11'
	]
	ppl_based_pmi_aggregation = ['max', 'mean', 'min', 'index_0', 'index_1']
#	export_prefix = 'experiment.scoring_comparison'
	no_inference = False
	for _model, _max_length, _batch in models:
		for scoring_method in methods:
			if 'gpt' in _model and scoring_method in methods_mlm:
				continue
			scorer = alm.RelationScorer(model=_model, max_length=_max_length, ckpt_path=ckpt_path, gpu_id=args.gpu_id, cache_dir=args.cache_dir)
			for _data in data:
				for _temp in all_templates:
					shared = dict(
						scoring_method=scoring_method,
						data=_data,
						template_type=_temp,
						batch_size=_batch,
						export_dir=args.export_dir,
						export_prefix=export_prefix,
						no_inference=no_inference,
						positive_permutation_aggregation=positive_permutation_aggregation,
						ppl_based_pmi_aggregation=ppl_based_pmi_aggregation
					)
					if scoring_method == 'pmi_feldman':
						for i in pmi_feldman_aggregation:
							scorer.analogy_test(pmi_feldman_aggregation=i, **shared)
							scorer.release_cache()
					else:
						scorer.analogy_test(**shared)
					scorer.release_cache()
	alm.export_report(export_prefix=export_prefix, export_dir=args.export_dir)

if not SKIP_TEST:
	logging.info('#######################################################################')
	logging.info('# get test accuracy on each combination of model and scoring function #')
	logging.info('#######################################################################')
	no_inference = False
#	methods = ['pmi_feldman', 'embedding_similarity', 'ppl', 'ppl_based_pmi', 'ppl_marginal_bias',
#			   'ppl_hypothesis_bias']
#	methods += ['ppl_head_masked', 'ppl_tail_masked', 'ppl_add_masked']
	methods = ['embedding_similarity', 'ppl']
#	export_prefix = 'experiment.scoring_comparison'
	df = alm.get_report(export_prefix=export_prefix, export_dir=args.export_dir)
	for i, m, s in product(data, models, methods):

		_model, _len, _batch = m
		if 'gpt' in _model and s in methods_mlm:
			continue
		"""
		if 'sph' in _model:
			tag = _model[3:]
		else:
			tag = _model
		"""
		tmp_df = df[df.data == i]
		tmp_df = tmp_df[tmp_df.model == _model]
		tmp_df = tmp_df[tmp_df.scoring_method == s]
		val_accuracy = tmp_df.sort_values(by='accuracy', ascending=False).head(1)['accuracy'].values[0]
		logging.info("RUN TEST:\n - data: {} \n - lm: {} \n - score: {} \n - validation accuracy: {} ".format(
			i, _model, s, val_accuracy))
		best_configs = tmp_df[tmp_df['accuracy'] == val_accuracy]
		logging.info("find {} configs with same accuracy".format(len(best_configs)))
		for n, tmp_df in best_configs.iterrows():
			config = json.loads(tmp_df.to_json())
			config.pop('accuracy')
			config.pop('max_length')
			scorer = alm.RelationScorer(model=config.pop('model'), max_length=_len, ckpt_path=ckpt_path, gpu_id=args.gpu_id, cache_dir=args.cache_dir)
			scorer.analogy_test(
				no_inference=no_inference,
				test=True,
				export_dir=args.export_dir,
				export_prefix=export_prefix,
				batch_size=_batch,
				val_accuracy=val_accuracy,
				**config
			)
			scorer.release_cache()
	alm.export_report(export_prefix=export_prefix, export_dir=args.export_dir, test=True)

