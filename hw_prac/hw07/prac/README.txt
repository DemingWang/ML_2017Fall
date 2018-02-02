Commands to run the program:

1. The main moudule of multiclass_perc.py looks like this:

# tuning
# run_tune_epochs()
run_tune_d()
# run_tune_sigma()

# testing
# test_mperceptron()
# test_mkperceptron_linear()
# test_mkperceptron_poly()
# test_mkperceptron_gau()


To get outputs of tuning for number of epochs we first 
comment other function (as shown above) and run the python command:
python3 multiclass_perc.py > outputs/tune_epochs.txt


similarly we create other output files.
