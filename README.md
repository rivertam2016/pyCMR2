# pyCMR2

Python implementation of CMR2 matlab code from:

Lohnas, L. J., Polyn, S. M., & Kahana, M. J. (2015). 
Expanding the scope of memory search: Modeling intralist and interlist effects in free recall. 
Psychological review, 122(2), 337-363.

The K02_files directory contains:

1. K02_data.txt: A .txt file containing the presented lists used in Kahana et al. (2002)

2. K02_LSA.txt:  A .txt file containing LSA cos-theta values (inter-item similarities) for these stimuli

3. K02_recs.txt: A .txt file containing the recall outputs from Kahana et al. (2002) 

4. K02_subject_ids.txt: A .txt file to break up K02_data.txt into lists belonging to each subject

The pyCMR2 directory contains:

1. pyCMR2:         A python version of the CMR2 code used in Lohnas et al. (2015)

2. runCMR2.py:     A python script enabling a basic launch of pyCMR2

5. graph_CMR2.py:  Code to reproduce the SPC and PFR curves from Figure 2 in Lohnas et al. (2015).

The K02 files, as well as the original MATLAB version of this code, are all available 
on the website of the Computational Memory Lab at UPenn.

Faster / more streamlined implementations of this code are in the pipeline. Stay tuned!
