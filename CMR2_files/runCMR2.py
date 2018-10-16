import pyCMR2 as CMR2
import numpy as np
import os

def main():
    """
    Set data & LSA paths pointing to a data and LSA matrix, respectively.

    In the Kahana et al. (2002) data (i.e., the K02 files), all subjects'
    sessions are concatenated into one file. The code is built to handle
    files in this way and separates them according to the indices obtained
    from the K02_subject_ids.txt file.

    There is some scratch code to try to work toward handling files in other
    formats, but don't use it yet. i.e. it is not operational.

    """

    # desired model parameters
    params_K02 = {

        'beta_enc': 0.519769,           # rate of context drift during encoding
        'beta_rec': 0.627801,           # rate of context drift during recall
        'beta_rec_post': 0.802543,      # rate of context drift between lists
                                        # (i.e., post-recall)

        'gamma_fc': 0.425064,  # learning rate, feature-to-context
        'gamma_cf': 0.895261,  # learning rate, context-to-feature
        'scale_fc': 1 - 0.425064,
        'scale_cf': 1 - 0.895261,

        's_cf': 1.292411,       # scales influence of semantic similarity
                                # on M_CF matrix

        's_fc': 0.0,            # scales influence of semantic similarity
                                # on M_FC matrix.
                                # s_fc is first implemented in
                                # Healey et al. 2016;
                                # set to 0.0 for prior papers.

        'phi_s': 1.408899,      # primacy parameter
        'phi_d': 0.989567,      # primacy parameter

        'nlists_for_accumulator': 4,    # only the list-length * 4
                                        # strongest items compete

        'kappa': 0.312686,      # item activation decay rate
        'lamb': 0.12962,        # lateral inhibition

        'eta': 0.392847,        # width of noise gaussian

        'rec_time_limit': 30000,  # ms allowed for recall post-study list
        'dt': 10.0,               # time scale for leaky accumulator
        'dt_tau': 0.01,
        'sq_dt_tau': 0.1000,

        'c_thresh': 0.073708,   # threshold of context similarity
                                # required for an item to be recalled

        'omega': 11.894106,     # repetition prevention parameter
        'alpha': 0.678955,      # repetition prevention parameter
    }

    # set data and LSA matrix paths
    LSA_path = '../K02_files/K02_LSA.txt'
    LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)

    data_path = '../K02_files/K02_data.txt'
    subjects_path = '../K02_files/K02_subject_ids.txt'

    # run CMR2 on the data
    resp, times = CMR2.run_CMR2(
            LSA_mat=LSA_mat, data_path=data_path,
            params=params_K02, subj_id_path=subjects_path, sep_files=False)

    # make a directory in which to save the output
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # save the output
    np.savetxt('./output/CMR2_recnos_K02.txt',  resp, delimiter=',', fmt='%.0d')
    np.savetxt('./output/CMR2_times_K02.txt', times, delimiter=',', fmt='%.0d')

if __name__ == "__main__":
    main()