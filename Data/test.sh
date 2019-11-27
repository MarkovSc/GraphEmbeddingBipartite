python experiment.py --lam 0.01 --max-iter 100 --model-name wiki --lip 1 --large 2 --gamma 1 --vectors-u wiki/SDNE/vectors_u.dat --vectors-v wiki/SDNE/vectors_v.dat --case-train wiki/case_train.dat --case-test wiki/case_test.dat
python experiment.py --lam 0.01 --max-iter 100 --model-name wiki --lip 1 --large 2 --gamma 1 --vectors-u wiki/BINE/vectors_u.dat --vectors-v wiki/BINE/vectors_v.dat --case-train wiki/case_train.dat --case-test wiki/case_test.dat
python experiment.py --lam 0.01 --max-iter 100 --model-name wiki --lip 1 --large 2 --gamma 1 --vectors-u wiki/EigenMap/vectors_u.dat --vectors-v wiki/EigenMap/vectors_v.dat --case-train wiki/case_train.dat --case-test wiki/case_test.dat
