@REM Sensitivity testing for stochastic BIM
python tests.py --samples 10 --bim --obj_weights 5 500 1500 3 10 --results_file alt_1a
python tests.py --samples 10 --bim --obj_weights 1 1000 1500 3 10 --results_file alt_1b
python tests.py --samples 10 --bim --obj_weights 1 500 2500 3 10 --results_file alt_1c
python tests.py --samples 10 --bim --obj_weights 1 500 1500 10 10 --results_file alt_1d
python tests.py --samples 10 --bim --obj_weights 1 500 1500 3 50 --results_file alt_1e
python tests.py --samples 10 --bim --n_emerg_lambda 8 --results_file alt_2a
python tests.py --samples 10 --bim --n_emerg_lambda 16 --results_file alt_2b
python tests.py --samples 10 --bim --elective_family_nums 8 4 2 2 --results_file alt_3a
python tests.py --samples 10 --bim --elective_family_nums 4 8 4 2 --results_file alt_3b
python tests.py --samples 10 --bim --elective_family_nums 4 4 8 2 --results_file alt_3c
python tests.py --samples 10 --bim --elective_family_nums 4 4 2 8 --results_file alt_3d
python tests.py --samples 10 --bim --n_rooms 3 --results_file alt_4a
python tests.py --samples 10 --bim --n_rooms 7 --results_file alt_4b

@REM BIM vs no BIM at start and end of day (stochastic electives only)
python tests.py --samples 10 --bim --results_file stoch_el_bim
python tests.py --samples 10 --bim --results_file stoch_el_bim --read
python tests.py --samples 10 --results_file stoch_el_no_bim
python tests.py --samples 10 --results_file stoch_el_no_bim --read

@REM Stochastic vs deterministic electives at start and end of day (BIM only)
@REM python tests.py --samples 10 --bim --results_file stoch_el_bim
@REM python tests.py --samples 10 --bim --results_file stoch_el_bim --read
python tests.py --det_el --bim --results_file det_el_bim
python tests.py --det_el --bim --results_file det_el_bim --read

@REM Oracle vs stochastic at end of day (BIM only)
@REM python tests.py --samples 10 --bim --results_file stoch_el_bim
@REM python tests.py --samples 10 --bim --results_file stoch_el_bim --read
python tests.py --det_el --oracle --bim --results_file oracle_bim
python tests.py --det_el --oracle --bim --results_file oracle_bim --read
