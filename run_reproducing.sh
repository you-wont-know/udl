### Main experiment (comparing acquisition functions):
python main.py --seed 1 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 1 --acq bald_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 1 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 1 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 1 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores

python main.py --seed 2 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 2 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 2 --acq bald_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 2 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 2 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores

python main.py --seed 3 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 3 --acq bald_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 3 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 3 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --seed 3 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores


### Deterministic model/acquisition function experiment:
python main.py --deterministic --seed 1 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 1 --acq bald_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 1 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 1 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 1 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores

python main.py --deterministic --seed 2 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 2 --acq bald_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 2 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 2 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 2 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores

python main.py --deterministic --seed 3 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 3 --acq bald_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 3 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 3 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --deterministic --seed 3 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores
