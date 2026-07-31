README: ESS8 CCA-ready belief datasets
=======================================

Purpose
-------
This curation reproduces the collaborator-approved ESS Round 8 preprocessing
for the CCA/network stage and provides separate files with and without the
official ESS survey-weight columns.

Raw input
---------
File: ESS8e02_3.csv
Raw respondents: 44387
Raw countries: 23

Final sample
------------
CCA-ready respondents: 37118
Countries: 23
Belief variables: 20

Sample rule
-----------
Respondents are retained if age is at least 18 or missing and no more than
two of the 20 constructed belief variables are missing.

Files
-----
1. ess8_cca_initial_beliefs_without_weights.csv
   Contains identifiers, country information, demographics, simple voting
   metadata, 20 belief variables, and belief-missingness counts. It contains
   no survey-weight columns.

2. ess8_cca_initial_beliefs_with_weights.csv
   Contains the same respondents and values as the without-weights file, plus
   dweight, pspwght, pweight, and anweight copied directly from the raw ESS
   file.

3. ess8_belief_validation_against_paper.csv
   Compares reproduced N, mean, and SD with Supplementary Table A2.

Official ESS weights
--------------------
dweight: design weight.
pspwght: post-stratification weight including the design weight.
pweight: population-size weight for cross-national aggregation.
anweight: ESS analysis weight supplied in the raw file.

The official weights are not recalculated. They are not belief variables and
must not be used as nodes in CCA or belief-network inference. The belief values
are not multiplied by the weights, and rows are not replicated.

Validation
----------
The final sample contains 37,118 respondents from 23 countries. All 20
belief-variable Ns match Supplementary Table A2 exactly; means and standard
deviations match when rounded to two decimal places. Removing the four weight
columns from the with-weights file yields the without-weights file exactly.
