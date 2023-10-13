@REM Sensitivity testing for stochastic BIM
python311 tests.py --samples 10 --bim --obj_weights 5 500 1500 3 10 --results_file alt_1a
python311 tests.py --samples 10 --bim --obj_weights 1 1000 1500 3 10 --results_file alt_1b
python311 tests.py --samples 10 --bim --obj_weights 1 500 2500 3 10 --results_file alt_1c
python311 tests.py --samples 10 --bim --obj_weights 1 500 1500 10 10 --results_file alt_1d
python311 tests.py --samples 10 --bim --obj_weights 1 500 1500 3 50 --results_file alt_1e
python311 tests.py --samples 10 --bim --n_emerg_lambda 8 --results_file alt_2a
python311 tests.py --samples 10 --bim --n_emerg_lambda 16 --results_file alt_2b
python311 tests.py --samples 10 --bim --elective_family_nums 8 4 2 2 --results_file alt_3a
python311 tests.py --samples 10 --bim --elective_family_nums 4 8 4 2 --results_file alt_3b
python311 tests.py --samples 10 --bim --elective_family_nums 4 4 8 2 --results_file alt_3c
python311 tests.py --samples 10 --bim --elective_family_nums 4 4 2 8 --results_file alt_3d
python311 tests.py --samples 10 --bim --n_rooms 3 --results_file alt_4a
python311 tests.py --samples 10 --bim --n_rooms 7 --results_file alt_4b

@REM BIM vs no BIM at start and end of day (stochastic electives only)
python311 tests.py --samples 10 --bim --results_file stoch_el_bim
python311 tests.py --samples 10 --results_file stoch_el_no_bim

@REM Stochastic vs deterministic electives at start and end of day (BIM only)
@REM python311 tests.py --samples 10 --bim --results_file stoch_el_bim
python311 tests.py --det_el --bim --results_file det_el_bim

@REM Oracle vs stochastic at end of day (BIM only)
@REM python311 tests.py --samples 10 --bim --results_file stoch_el_bim
python311 tests.py --det_el --oracle --bim --results_file oracle_bim
