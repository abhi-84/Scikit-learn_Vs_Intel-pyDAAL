# Scikit-learn_Vs_Intel-pyDAAL
# Three popular Machine Learning algorithms such as KNN, Naive-Bayes and SVM are compared here using Scikit-learn and Intel pyDAAL library.
# For comparison, i have used Jupyter notebook and magic function "%timeit" which gives "best" execution times in "ms" or "us".
# It is clearly visible from "test" results that Intel DAAL library performs better for KNN and SVM algorithm when compared with Scikit-learn library! 

K_nearest_neighbours_train Database

KNN (Scikit-learn)

train
9.2 ms ± 105 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

test
723 ms ± 15.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

F1 Score
0.9734375287390278


KNN (pyDAAL)--Using IDP
train
40.5 ms ± 597 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

test
25.2 ms ± 1.33 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)




Naive-Bayes (Scikit-learn)

train
6.76 ms ± 53.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


test
5.42 ms ± 27.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

F1 Score
1.0


Naive-Bayes (pyDAAL) – using IDP

train
26 ms ± 544 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

test
6.51 ms ± 99.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



SVM (Scikit-learn) 
train
4.33 ms ± 106 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

test
2.09 ms ± 93.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

F1 Score
0.9754684599488634


SVM (pyDAAL) – using IDP

train
5.38 ms ± 117 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

test
836 µs ± 13.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

