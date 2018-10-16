# pyCMR2

This project contains a Python 3 implementation of CMR2 matlab code from the paper:

Lohnas, L. J., Polyn, S. M., & Kahana, M. J. (2015). 
Expanding the scope of memory search: Modeling intralist and interlist effects in free recall. 
Psychological review, 122(2), 337-363.

<b>The K02_files directory contains:</b>

1. K02_data.txt: The presented lists used in Kahana et al. (2002)

2. K02_LSA.txt:  The LSA cos-theta values (inter-item similarities) for these stimuli

3. K02_recs.txt: The recall outputs from Kahana et al. (2002) 

4. K02_subject_ids.txt: Indices to break up K02_data.txt into the lists belonging to each subject

<b>The CMR2_files directory contains:</b>

1. pyCMR2:         A python version of the CMR2 code used in Lohnas et al. (2015)

2. runCMR2.py:     A python script enabling a basic launch of pyCMR2

5. graph_CMR2.py:  Code to reproduce the SPC and PFR curves from Figure 2 in Lohnas et al. (2015).

<b> Additional notes: </b>

The K02 data files and the original MATLAB version of this code are available 
on the website of the Computational Memory Lab at the University of Pennsylvania.

Faster / more streamlined implementations of this code are in the pipeline. Stay tuned!
