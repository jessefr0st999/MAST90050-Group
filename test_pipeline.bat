python tests.py --samples 10 --bim --results_file stoch_el_bim --heur_log --jobs_file default_jobs
python tests.py --samples 10 --results_file stoch_el_no_bim --heur_log --jobs_file default_jobs --read_jobs

python tests.py --samples 10 --n_emerg_lambda 8 --bim --elective_family_nums 1 2 3 4 --results_file stoch_el_bim_1234 --heur_log --jobs_file default_jobs_1234 
python tests.py --samples 10 --n_emerg_lambda 8 --elective_family_nums 1 2 3 4 --results_file stoch_el_no_bim_1234 --heur_log --jobs_file default_jobs_1234 --read_jobs

python tests.py --samples 10 --n_emerg_lambda 12 --bim --elective_family_nums 3 4 1 2 --results_file stoch_el_bim_3412 --heur_log --jobs_file default_jobs_3412 
python tests.py --samples 10 --n_emerg_lambda 12 --elective_family_nums 3 4 1 2 --results_file stoch_el_no_bim_3412 --heur_log --jobs_file default_jobs_3412 --read_jobs

python tests.py --samples 10 --bim --obj_weights 5 500 1500 3 10 --results_file alt_1a --jobs_file default_jobs --read_jobs
python tests.py --samples 10 --bim --obj_weights 1 1000 1500 3 10 --results_file alt_1b --jobs_file default_jobs --read_jobs
python tests.py --samples 10 --bim --obj_weights 1 500 2500 3 10 --results_file alt_1c --jobs_file default_jobs --read_jobs
python tests.py --samples 10 --bim --obj_weights 1 500 1500 10 10 --results_file alt_1d --jobs_file default_jobs --read_jobs
python tests.py --samples 10 --bim --obj_weights 1 500 1500 3 50 --results_file alt_1e --jobs_file default_jobs --read_jobs

python tests.py --samples 10 --bim --n_emerg_lambda 5 --results_file alt_2a --jobs_file alt_2a
python tests.py --samples 10 --bim --n_emerg_lambda 15 --results_file alt_2b --jobs_file alt_2b

python tests.py --samples 10 --bim --elective_family_nums 7 1 1 1 --results_file alt_3a --jobs_file alt_3a
python tests.py --samples 10 --bim --elective_family_nums 1 7 1 1 --results_file alt_3b --jobs_file alt_3b
python tests.py --samples 10 --bim --elective_family_nums 1 1 7 1 --results_file alt_3c --jobs_file alt_3c
python tests.py --samples 10 --bim --elective_family_nums 1 1 1 7 --results_file alt_3d --jobs_file alt_3d
@REM python tests.py --samples 10 --bim --elective_family_nums 5 4 3 2 --results_file alt_3e --jobs_file alt_3e
python tests.py --samples 10 --bim --elective_family_nums 3 2 1 0 --results_file alt_3f --jobs_file alt_3f

python tests.py --samples 10 --bim --rooms 3 --results_file alt_4a --jobs_file default_jobs --read_jobs
python tests.py --samples 10 --bim --rooms 7 --results_file alt_4b --jobs_file default_jobs --read_jobs

python tests.py --samples 10 --bim --obj_weights 0 500 1500 3 10 --results_file alt_5a --jobs_file default_jobs --read_jobs
python tests.py --samples 10 --bim --obj_weights 1 0 1500 3 10 --results_file alt_5b --jobs_file default_jobs --read_jobs
python tests.py --samples 10 --bim --obj_weights 1 500 0 3 10 --results_file alt_5c --jobs_file default_jobs --read_jobs
python tests.py --samples 10 --bim --obj_weights 1 500 1500 0 10 --results_file alt_5d --jobs_file default_jobs --read_jobs

python tests.py --samples 10 --bim --n_emerg_lambda 5 --elective_family_nums 3 2 1 0 --rooms 3 --results_file alt_6a --jobs_file alt_6a
python tests.py --samples 10 --bim --elective_family_nums 5 4 3 2 --results_file alt_3e --jobs_file alt_3e
python tests.py --samples 10 --bim --n_emerg_lambda 15 --elective_family_nums 5 4 3 2 --rooms 7 --results_file alt_6b --jobs_file alt_6b