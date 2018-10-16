import numpy as np
import matplotlib.pyplot as plt
import os

##########
#
#   Define some helper methods
#
##########

def recode_for_spc(data_recs, data_pres):
    """Helper method to recode data for an spc curve"""
    ll = data_pres.shape[1]
    maxlen = ll * 2

    rec_lists = []
    for i in range(len(data_recs)):
        this_list = data_recs[i]
        pres_list = data_pres[i]

        this_list = this_list[this_list > 0]

        # get indices of first place each unique value appears
        indices = np.unique(this_list, return_index=True)[1]

        # get each unique value in array (by first appearance)
        this_list_unique = this_list[sorted(indices)]

        # get the indices of these values in the other list, and add 1
        list_recoded = np.nonzero(this_list_unique[:, None] == pres_list)[1] + 1

        # re-pad with 0's so we can reformat this as a matrix again later
        recoded_row = np.pad(list_recoded, pad_width=(
            0, maxlen - len(list_recoded)),
                             mode='constant', constant_values=0)

        # append to running list of recoded rows
        rec_lists.append(recoded_row)

    # reshape as a matrix
    recoded_lists = np.asmatrix(rec_lists)

    return recoded_lists


def get_spc_pfc(rec_lists, ll):
    """Get spc and pfc for the recoded lists"""

    spclists = []
    pfclists = []
    for each_list in rec_lists:

        each_list = each_list[each_list > 0]

        # init. list to store whether or not an item was recalled
        spc_counts = np.zeros((1, ll))
        pfc_counts = np.zeros((1, ll))

        # get indices of where to put items in the list;
        # items start at 1, so index needs to -1
        spc_count_indices = each_list - 1
        spc_counts[0, spc_count_indices] = 1

        if each_list.shape[1] <= 0:
            continue
        else:
            # get index for first item in list
            pfc_count_index = each_list[0, 0] - 1
            pfc_counts[0, pfc_count_index] = 1

            spclists.append(np.squeeze(spc_counts))
            pfclists.append(np.squeeze(pfc_counts))

    # if no items were recalled, output a matrix of 0's
    if not spclists:
        spcmat = np.zeros((rec_lists.shape[0], ll))
    else:
        spcmat = np.array(spclists)

    if not pfclists:
        pfcmat = np.zeros((rec_lists.shape[0], ll))
    else:
        pfcmat = np.array(pfclists)

    # get mean and sem's for spc and pfc
    spc_mean = np.nanmean(spcmat, axis=0)
    spc_sem  = np.nanstd(spcmat, axis=0) / (len(spcmat) ** 0.5)

    pfc_mean = np.nanmean(pfcmat, axis=0)
    pfc_sem  = np.nanstd(pfcmat, axis=0) / (len(pfcmat) ** 0.5)

    return spc_mean, spc_sem, pfc_mean, pfc_sem


def main():

    """Graph some results from previously generated output (SPC, PFC)"""

    # set list length here
    ll = 10

    # decide whether to save figs out or not
    save_figs = True

    ###############
    #
    #   Get data output
    #
    ###############

    # set paths
    data_path = '../K02_files/K02_data.txt'
    data_rec_path = '../K02_files/K02_recs.txt'
    data_pres = np.loadtxt(data_path, delimiter=',')
    data_rec = np.loadtxt(data_rec_path, delimiter=',')

    # recode data lists for spc and pfc analyses
    recoded_lists = recode_for_spc(data_rec, data_pres)

    # save out the recoded lists in case you want to read this in later
    np.savetxt('./output/data_recoded.txt', recoded_lists,
               delimiter=',', fmt='%.0d')

    # get spc & pfc
    target_spc, target_spc_sem, target_pfc, target_pfc_sem = \
        get_spc_pfc(recoded_lists, ll)

    ###############
    #
    #   Get CMR2 output
    #
    ###############

    rec_nos = np.loadtxt('./output/CMR2_recnos_K02.txt', delimiter=',')
    cmr_recoded_output = recode_for_spc(rec_nos, data_pres)

    # get the model's spc and pfc predictions:
    (this_spc, this_spc_sem, this_pfc,
    this_pfc_sem) = get_spc_pfc(cmr_recoded_output, ll)

    print("\nData vals: ")
    print(target_spc)
    print(target_pfc)

    print("\nModel vals: ")
    print(this_spc)
    print(this_pfc)

    ###############
    #
    #   Plot graphs
    #
    ###############

    # make a directory in which to save the figures
    figs_dir = 'Figs'
    if not os.path.exists(figs_dir):
        os.mkdir(figs_dir)

    # line width
    lw = 2

    # gray color for CMR predictions
    gray = '0.50'

    #_______________________ plot spc
    plt.figure()
    xvals = range(1, ll+1, 1)     # ticks for x-axis

    plt.plot(xvals, this_spc, lw=lw, c=gray, linestyle='--', label='CMR2')
    plt.plot(xvals, target_spc, lw=lw, c='k', label='Data')

    plt.ylabel('Probability of Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, ll+.5, 0, 1], size='large')
    plt.legend(loc='upper left')
    plt.title('Serial Position Curve', size='large')

    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/spc_fig.eps', format='eps', dpi=1000)

    #_______________________ plot pfc
    plt.figure()
    plt.plot(xvals, this_pfc, lw=lw, c=gray, linestyle='--', label='CMR2')
    plt.plot(xvals, target_pfc, lw=lw, c='k', label='Data')

    plt.title('Probability of First Recall', size='large')
    plt.xlabel('Serial Position', size='large')
    plt.ylabel('Probability of First Recall', size='large')
    plt.xticks(xvals, size='large')
    plt.yticks(np.arange(0.0, 1.2, 0.2), size='large')
    plt.axis([0.5, ll+.5, 0, 1], size='large')
    plt.legend(loc='upper left')

    # save fig nicely
    if save_figs:
        plt.savefig('./Figs/pfc_fig.eps', format='eps', dpi=1000)

    plt.show()


if __name__ == "__main__":
    main()


