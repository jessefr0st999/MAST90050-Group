@REM Stochastic vs deterministic electives at start and end of day (BIM only)
@REM python311 tests.py --samples 10 --bim --results_file stoch_el_bim --heur_log
python311 tests.py --det_el --bim --results_file det_el_bim --heur_log

@REM Oracle vs stochastic at end of day (BIM only)
@REM python311 tests.py --samples 10 --bim --results_file stoch_el_bim --heur_log
python311 tests.py --det_el --oracle --bim --results_file oracle_bim --heur_log

@REM BIM vs no BIM at start and end of day (stochastic electives only)
python311 tests.py --samples 10 --bim --results_file stoch_el_bim --heur_log
python311 tests.py --samples 10 --results_file stoch_el_no_bim --heur_log

@REM Sensitivity testing for stochastic BIM
python311 tests.py --samples 10 --bim --obj_weights 5 500 1500 3 10 --results_file alt_1a
python311 tests.py --samples 10 --bim --obj_weights 1 1000 1500 3 10 --results_file alt_1b
python311 tests.py --samples 10 --bim --obj_weights 1 500 2500 3 10 --results_file alt_1c
python311 tests.py --samples 10 --bim --obj_weights 1 500 1500 10 10 --results_file alt_1d
python311 tests.py --samples 10 --bim --obj_weights 1 500 1500 3 50 --results_file alt_1e
python311 tests.py --samples 10 --bim --n_emerg_lambda 5 --results_file alt_2a
python311 tests.py --samples 10 --bim --n_emerg_lambda 15 --results_file alt_2b
python311 tests.py --samples 10 --bim --elective_family_nums 9 3 2 1 --results_file alt_3a
python311 tests.py --samples 10 --bim --elective_family_nums 4 8 2 1 --results_file alt_3b
python311 tests.py --samples 10 --bim --elective_family_nums 4 3 7 1 --results_file alt_3c
python311 tests.py --samples 10 --bim --elective_family_nums 4 3 2 6 --results_file alt_3d
python311 tests.py --samples 10 --bim --rooms 3 --results_file alt_4a
python311 tests.py --samples 10 --bim --rooms 7 --results_file alt_4b