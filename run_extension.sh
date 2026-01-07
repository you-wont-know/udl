### Two-stage experiment:
python main.py --seed 1 --acq random_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 1 --acq bald_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 1 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 1 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 1 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores_two_stage

python main.py --seed 2 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 2 --acq random_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 2 --acq bald_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 2 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 2 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores_two_stage

python main.py --seed 3 --acq random_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 3 --acq bald_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 3 --acq predictive_entropy_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 3 --acq variation_ratio_acquisition --pool-scores evaluate_pool_scores_two_stage
python main.py --seed 3 --acq mean_standard_deviation_acquisition --pool-scores evaluate_pool_scores_two_stage


### AI vs MFVI experiment:
python main.py --model analytical --sigma 1.0 --seed 1 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 1 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 1 --acq trace_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 1 --acq trace_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 1 --acq determinant_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 1 --acq determinant_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 1 --acq bayesian_variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 1 --acq bayesian_variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 1 --acq bayesian_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 1 --acq bayesian_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 1 --acq max_diag_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 1 --acq max_diag_acquisition --pool-scores evaluate_pool_scores

python main.py --model analytical --sigma 1.0 --seed 2 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 2 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 2 --acq trace_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 2 --acq trace_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 2 --acq determinant_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 2 --acq determinant_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 2 --acq bayesian_variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 2 --acq bayesian_variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 2 --acq bayesian_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 2 --acq bayesian_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 2 --acq max_diag_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 2 --acq max_diag_acquisition --pool-scores evaluate_pool_scores

python main.py --model analytical --sigma 1.0 --seed 3 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 3 --acq random_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 3 --acq trace_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 3 --acq trace_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 3 --acq determinant_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 3 --acq determinant_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 3 --acq bayesian_variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 3 --acq bayesian_variation_ratio_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 3 --acq bayesian_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 3 --acq bayesian_entropy_acquisition --pool-scores evaluate_pool_scores
python main.py --model analytical --sigma 1.0 --seed 3 --acq max_diag_acquisition --pool-scores evaluate_pool_scores
python main.py --model variational --sigma 1.0 --seed 3 --acq max_diag_acquisition --pool-scores evaluate_pool_scores
