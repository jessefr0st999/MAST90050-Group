@REM Deterministic electives
python311 tests.py --det_el --bim --results_file det_el_bim --heur_log
python311 tests.py --det_el --results_file det_el_no_bim --heur_log
python311 tests.py --det_el --oracle --results_file oracle --heur_log

@REM Stochastic electives (BIM and no BIM)
python311 tests.py --samples 10 --bim --results_file stoch_el_bim --heur_log --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --results_file stoch_el_no_bim --heur_log --jobs_file default_jobs --read_jobs

@REM Sensitivity testing for stochastic BIM
python311 tests.py --samples 10 --bim --obj_weights 5 500 1500 3 10 --results_file alt_1a --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --obj_weights 1 1000 1500 3 10 --results_file alt_1b --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --obj_weights 1 500 2500 3 10 --results_file alt_1c --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --obj_weights 1 500 1500 10 10 --results_file alt_1d --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --obj_weights 1 500 1500 3 50 --results_file alt_1e --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --n_emerg_lambda 5 --results_file alt_2a --jobs_file alt_2a
python311 tests.py --samples 10 --bim --n_emerg_lambda 15 --results_file alt_2b --jobs_file alt_2b
python311 tests.py --samples 10 --bim --elective_family_nums 9 3 2 1 --results_file alt_3a --jobs_file alt_3a
python311 tests.py --samples 10 --bim --elective_family_nums 4 8 2 1 --results_file alt_3b --jobs_file alt_3b
python311 tests.py --samples 10 --bim --elective_family_nums 4 3 7 1 --results_file alt_3c --jobs_file alt_3c
python311 tests.py --samples 10 --bim --elective_family_nums 4 3 2 6 --results_file alt_3d --jobs_file alt_3d
python311 tests.py --samples 10 --bim --rooms 3 --results_file alt_4a --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --rooms 7 --results_file alt_4b --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --obj_weights 0 500 1500 3 10 --results_file alt_5a --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --obj_weights 1 0 1500 3 10 --results_file alt_5b --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --obj_weights 1 500 0 3 10 --results_file alt_5c --jobs_file default_jobs --read_jobs
python311 tests.py --samples 10 --bim --obj_weights 1 500 1500 0 10 --results_file alt_5d --jobs_file default_jobs --read_jobs