"""This code implements a python version of the context maintenance and
retrieval model from Lohnas et al. (2015). Please email Rivka Cohen at
rivkat.cohen@gmail.com with questions or bug reports!"""

# when hooked up to some libraries, numpy.dot() uses multithreading.
# This will cause problems for you if you try to distribute on a cluster,
# so turn this off with mkl.
import mkl
mkl.set_num_threads(1)
import numpy as np
import scipy.io
import math
from glob import glob
import time


def norm_vec(vec):
    """Helper method to normalize a vector"""

    # get the square root of the sum of each element in the dat vector squared
    denom = np.sqrt(np.sum(vec**2))

    # if the resulting denom value equals 0.0, then set this equal to 1.0
    if denom == 0.0:
        return vec
    else:
        # divide each element in the vector by this denom
        return vec/denom


def advance_context(c_in_normed, c_temp, this_beta):
    """Helper function to advance context"""

    # if row vector, force c_in to be a column vector
    if c_in_normed.shape[1] > 1:
        c_in_normed = c_in_normed.T
    assert(c_in_normed.shape[1] == 1)  # sanity check

    # if col vector, force c_temp to be a row vector
    if c_temp.shape[0] > 1:
        c_temp = c_temp.T
    assert(c_temp.shape[0] == 1)  # sanity check

    # calculate rho
    rho = (math.sqrt(1 + (this_beta**2)*
                     ((np.dot(c_temp, c_in_normed)**2) - 1)) -
           this_beta*np.dot(c_temp, c_in_normed))

    # update context
    updated_c = rho*c_temp + this_beta * c_in_normed.T

    # send updated_c out as a col vector
    if updated_c.shape[1] > 1:
        updated_c = updated_c.T

    return updated_c


class CMR2(object):
    """Initialize CMR2 class"""

    def __init__(self, params, LSA_mat, data_mat):
        """
        Initialize CMR2 object

        :param params: dictionary containing desired parameter values for CMR2
        :param LSA_mat: matrix containing LSA cos theta values between each item
            in the word pool.
        :param data_mat: matrix containing the lists of items that were
            presented to a given participant. Dividing this up is taken care
            of in the run_CMR2 method.
            You can also divide the data according to session, rather than
            by subject, if desired.  The run_CMR2 method is where you would
            alter this; simply submit sheets of presented items a session
            at a time rather than a subject at a time.

        ndistractors: There are 2x as many distractors as there are lists,
            because presenting a distractor is how we model the shift in context
            that occurs between lists.
            And, each list has a distractor that represents the distractor task.

            Additionally, an initial orthogonal item is presented prior to the
            first list, so that the system does
            not start with context as an empty 0 vector.

            In the weight matrices & context vectors, the distractors' region
            is located after study item indices.

        beta_in_play: The update_context_temp() method will always reference
            self.beta_in_play; beta_in_play changes between the
            different beta (context drift) values offered by the
            parameter set, depending on where we are in the simulation.
        """

        # Choose whether to allow weight matrices to update during retrieval
        self.learn_while_retrieving = False

        # data we are simulating output from
        self.pres_list_nos = data_mat.astype(np.int16)

        # data structure
        self.nlists = self.pres_list_nos.shape[0]
        self.listlength = self.pres_list_nos.shape[1]

        # total no. of study items presented to the subject in this session
        self.nstudy_items_presented = self.listlength * self.nlists

        # One distractor per list + 1 at the beginning to start the sys.
        self.ndistractors = self.nlists + 1

        # n cells in the temporal subregion
        self.templength = self.nstudy_items_presented + self.ndistractors

        # total number of elements operating in the system,
        # including all study lists and distractors
        self.nelements = (self.nstudy_items_presented + self.ndistractors)

        # make a list of all items ever presented to this subject & sort it
        self.all_session_items = np.reshape(self.pres_list_nos,
                                            (self.nlists*self.listlength))
        self.all_session_items_sorted = np.sort(self.all_session_items)

        # set parameters to those input when instantiating the model
        self.params = params

        #########
        #
        #   Make mini LSA matrix
        #
        #########

        # Create a mini-LSA matrix with just the items presented to this Subj.
        self.exp_LSA = np.zeros(
            (self.nstudy_items_presented, self.nstudy_items_presented),
                                dtype=np.float32)

        # Get list-item LSA indices
        for row_idx, this_item in enumerate(self.all_session_items_sorted):

            # get item's index in the larger LSA matrix
            this_item_idx = this_item - 1

            for col_idx, compare_item in enumerate(self.all_session_items_sorted):
                # get ID of jth item for LSA cos theta comparison
                compare_item_idx = compare_item - 1

                # get cos theta value between this_item and compare_item
                cos_theta = LSA_mat[this_item_idx, compare_item_idx]

                # place into this session's LSA cos theta matrix
                self.exp_LSA[int(row_idx), int(col_idx)] = cos_theta

        ##########
        #
        #   Set the rest of the variables we will need from the inputted params
        #
        ##########

        # beta used by update_context_temp(); more details in doc string
        self.beta_in_play = self.params['beta_enc']

        # init leaky accumulator vars
        self.steps = 0

        # track which study item has been presented
        self.study_item_idx = 0
        # track which list has been presented
        self.list_idx = 0

        # track which distractor item has been presented
        self.distractor_idx = self.nstudy_items_presented

        # list of items recalled throughout model run
        self.recalled_items = []

        ##########
        #
        #   Set up the learning rates for during retrieval.
        #   If you want different learning rates than during encoding, this is
        #   the place to change those.
        #   By default, these are set to the encoding rates.
        #
        ##########

        self.gamma_cf_retrieval = self.params['gamma_cf']
        self.gamma_fc_retrieval = self.params['gamma_fc']

        ##########
        #
        #   Set up & initialize weight matrices
        #
        ##########

        self.M_FC = np.identity(
            self.nelements, dtype=np.float32) * self.params['scale_fc']
        self.M_CF = np.identity(
            self.nelements, dtype=np.float32) * self.params['scale_cf']

        # set up / initialize feature and context layers
        self.c_net = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c_old = np.zeros((self.nelements, 1), dtype=np.float32)
        self.c_in_normed = np.zeros_like(self.c_net)
        self.f_net = np.zeros((self.nelements, 1), dtype=np.float32)

        # set up & track leaky accumulator & recall vars
        self.x_thresh_full = np.ones(
            self.nlists * self.listlength, dtype=np.float32)
        self.n_prior_recalls = np.zeros(
            [self.nstudy_items_presented, 1], dtype=np.float32)
        self.nitems_in_race = (self.listlength *
                               self.params['nlists_for_accumulator'])

        # track the list items that have been presented
        self.lists_presented = []

    def create_semantic_structure(self):
        """Layer semantic structure onto M_CF (and M_FC, if s_fc is nonzero)

        Dimensions of the LSA matrix for this subject are nitems x nitems.

        To get items' indices, we will subtract 1 from the item ID, since
        item IDs begin indexing at 1, not 0.
        """

        # scale the LSA values by s_cf, per Lohnas et al., 2015
        cf_exp_LSA = self.exp_LSA * self.params['s_cf'] * self.params['scale_cf']

        # add the scaled LSA cos theta values to NW quadrant of the M_CF matrix
        self.M_CF[:self.nstudy_items_presented, :self.nstudy_items_presented] \
            += cf_exp_LSA

        # scale the LSA values by s_fc, per Healey et al., 2016
        fc_exp_LSA = self.exp_LSA * self.params['s_fc'] * self.params['scale_fc']

        # add the scaled LSA cos theta values to NW quadrant of the M_FC matrix
        self.M_FC[:self.nstudy_items_presented, :self.nstudy_items_presented] \
            += fc_exp_LSA

    ################################################
    #
    #   Functions defining the recall process
    #
    ################################################

    def leaky_accumulator(self, in_act, x_thresholds, ncycles):
        """
        :param in_act: Top listlength * 4 item activations
        :param noise_vec: noise values for the accumulator.  These are
            calculated outside this function to save some runtime.
        :param x_thresholds: Threshold each item in the race must reach
            to be recalled. Starts at 1.0 for each item but can increase
            after an item is recalled in order to prevent repetitions.
        :return: Method returns index of the item that won the race,
            the time that elapsed for this item during the process of
            running the accumulator, and the final state of item activations,
            which although not strictly necessary, can be helpful for
            debugging.

        To prevent items from repeating, specify appropriate omega & alpha
        parameter values in the params dictionary.

        To facilitate testing changes to this model, you can comment out
        the noise_vec where it is added to x_s in the while loop below.
        """

        # get max number of accumulator cycles for this run of the accumulator
        nitems_in_race = in_act.shape[0]

        # track whether an item has passed the threshold
        item_has_crossed = False

        # initialize x_s vector with one 0 per accumulating item
        x_s = np.zeros(nitems_in_race, dtype=np.float32)

        # init counter at 1 instead of 0, for consistency with
        # Lynn & Sean's MATLAB code
        cycle_counter = 1

        # If no items were activated, don't bother running the accumulator.
        if np.sum(in_act) == 0:
            out_statement = (None, self.params['rec_time_limit'], x_s)
        else:
            # initialize list to track items that won the race
            rec_indices = []
            while cycle_counter < ncycles and not item_has_crossed:

                # we don't scale by eta the way Lynn & Sean do, because
                # np.random.normal has already scaled by eta for us.
                # Still have to scale by sq_dt_tau, though.

                noise_vec = np.random.normal(0, self.params['eta'], size=nitems_in_race)

                x_s = (x_s + (in_act - self.params['kappa']*x_s
                          - (sum(self.params['lamb']*x_s)
                             - (x_s*self.params['lamb'])))
                            * self.params['dt_tau']
                            + noise_vec * self.params['sq_dt_tau'])

                # In this implementation, accumulating evidence may not be negative.
                x_s[x_s < 0.0] = 0.0

                # if any item's evidence has passed its threshold,
                if np.any(x_s >= x_thresholds):
                    # let the system know that an item has crossed;
                    # stop accumulating.
                    item_has_crossed = True

                    # get the indices where items have passed their recall threshold
                    rec_indices = np.where(x_s >= x_thresholds)[0]

                cycle_counter += 1

            # calculate elapsed time:
            rt = cycle_counter * self.params['dt']

            # if no item has crossed,
            if rec_indices == []:
                winner_index = None

            # else, if an item has crossed:
            else:
                # if more than one item passed the threshold, pick one at random to
                # be the winner
                if len(rec_indices) > 1:
                    winner_index = np.random.choice(rec_indices)
                elif len(rec_indices) == 1:
                    winner_index = rec_indices[0]

            out_statement = (winner_index, rt, x_s)

        return out_statement

    ####################
    #
    #   Initialize and run a recall session
    #
    ####################

    def recall_session(self):
        """Simulate a recall portion of an experiment, following a list
        presentation. """

        time_passed = 0
        rec_time_limit = self.params['rec_time_limit']

        nlists_for_accumulator = self.params['nlists_for_accumulator']

        # set how many items get entered into the leaky accumulator process
        nitems_in_race = self.listlength * nlists_for_accumulator
        nitems_in_session = self.listlength * self.nlists

        # initialize list to store recalled items
        recalled_items = []
        RTs = []
        times_since_start = []

        # track & limit None responses to prevent a param vector
        # that yields a long series of None's from eating up runtime
        num_of_nones = 0

        # number of items allowed to recall beyond list length.
        # needed to constrain parameter fitting algorithms.
        num_extras = 3

        # run a recall session for the amount of time in rec_time_limit,
        # or until person recalls >= list length + num_extras,
        # or until person repeatedly recalls a "None" item too many times
        while ((time_passed < rec_time_limit)
               and (len(recalled_items) <= self.listlength + num_extras)
               and (num_of_nones <= self.listlength + num_extras)):

            # get item activations to input to the accumulator
            f_in = np.dot(self.M_CF, self.c_net)

            # sort f_in so we can get the ll * 4 items
            # with the highest activations
            sorted_indices = np.argsort(np.squeeze(f_in[:nitems_in_session]).T)
            sorted_activations = np.sort(f_in[:nitems_in_session].T)

            # get the top-40 activations, and the indices corresponding
            # to their position in the full list of presented items.
            # we need this second value to recover the item's ID later.
            in_activations = sorted_activations[0][
                             (nitems_in_session - nitems_in_race):]
            in_indices = sorted_indices[(nitems_in_session - nitems_in_race):]

            # determine max cycles for the accumulator.
            max_cycles = np.ceil(
                (rec_time_limit - time_passed) / self.params['dt'])

            # initialize the x_threshold vector
            x_thresh = self.x_thresh_full[in_indices]

            # get the winner of the leaky accumulator, its reaction time,
            # and this race's activation values.
            # x_n isn't strictly necessary, but can be helpful for debugging.

            winner_accum_idx, this_RT, x_n = self.leaky_accumulator(
                in_activations, x_thresh, max_cycles)

            # increment time counter
            time_passed += this_RT

            # If an item was retrieved, recover the item info corresponding
            # to the activation value index retrieved by the accumulator
            if winner_accum_idx is not None:

                # recover item's index from the original pool of item indices
                winner_sorted_idx = in_indices[winner_accum_idx]

                # get original item ID for this item
                winner_ID = np.sort(self.all_session_items)[winner_sorted_idx]

                ##########
                #
                #   Present item & then update system,
                #   as per regular article equations
                #
                ##########

                # reinstantiate this item
                self.present_item(winner_sorted_idx)

            # if no item was retrieved, instantiate a zero-vector
            # in the feature layer
            else:
                # track number of "None"s so we can stop the recall session
                # if we have drawn up blank more than ll + 3 times or so.
                num_of_nones += 1
                self.f_net = np.zeros([1, self.nelements])

            ##########
            #
            #   Whether or not the item is reported for recall,
            #   the item will still update the current context, as below.
            #
            ##########

            self.beta_in_play = self.params['beta_rec']
            self.update_context_temp()

            ############
            #
            #   See if the evoked context is similar enough to current context;
            #   If not, censor the item
            #
            ############

            # get similarity between c_old and the c retrieved by the item.
            c_similarity = np.dot(self.c_old.T, self.c_in_normed)

            # if sim threshold is passed,
            if (winner_accum_idx is not None) and (
                        c_similarity >= self.params['c_thresh']):

                # store item ID, RT, and time since start of rec session
                recalled_items.append(winner_ID)
                RTs.append(this_RT)
                times_since_start.append(time_passed)

                # Update the item's recall threshold & its prior recalls count
                self.x_thresh_full[winner_sorted_idx] = (
                    1 + self.params['omega'] * (
                        self.params['alpha'] ** self.n_prior_recalls[
                            winner_sorted_idx]))
                self.n_prior_recalls[winner_sorted_idx] += 1

                # if learning is enabled during recall process,
                # update the matrices
                if self.learn_while_retrieving:

                    # Update M_FC
                    M_FC_exp = np.dot(self.c_old, self.f_net)
                    self.M_FC += np.multiply(M_FC_exp, self.gamma_fc_retrieval)

                    # Update M_CF
                    M_CF_exp = np.dot(self.f_net.T, self.c_old.T)
                    self.M_CF += np.multiply(M_CF_exp, self.gamma_cf_retrieval)

            else:
                continue

        # update counter of what list we're on
        self.list_idx += 1

        return recalled_items, RTs, times_since_start

    def present_item(self, item_idx):
        """Set the f layer to a row vector of 0's with a 1 in the
        presented item location.  The model code will arrange this as a column
        vector where appropriate.

        :param: item_idx: index of the item being presented to the system"""

        # init feature layer vector
        self.f_net = np.zeros([1, self.nelements], dtype=np.float32)

        # code a 1 in the temporal region in that item's index
        self.f_net[0][item_idx] = 1

    def update_context_temp(self):
        """Updates the temporal region of the context vector.
        This includes all presented items, distractors, and the orthogonal
        initial item."""

        # get copy of the old c_net vector prior to updating it
        self.c_old = self.c_net.copy()

        # get input to the new context vector
        net_cin = np.dot(self.M_FC, self.f_net.T)

        # get nelements in temporal subregion
        nelements_temp = self.nstudy_items_presented + 2*self.nlists + 1

        # get region of context that includes all items presented
        # (items, distractors, & orthogonal initial item)
        cin_temp = net_cin[:nelements_temp]

        # norm the temporal region of the c_in vector
        cin_normed = norm_vec(cin_temp)
        self.c_in_normed[:nelements_temp] = cin_normed

        # grab the current values in the temp region of the network c vector
        net_c_temp = self.c_net[:nelements_temp]

        # get updated values for the temp region of the network c vector
        ctemp_updated = advance_context(
            cin_normed, net_c_temp, self.beta_in_play)

        # incorporate updated temporal region of c into the network's c vector
        self.c_net[:nelements_temp] = ctemp_updated

    def present_list(self):
        """
        Presents a single list of items.

        Update context using post-recall beta weight if distractor comes
        between lists.  Use beta_enc if distractor is the first item
        in the system; this item serves to init. context to non-zero values.

        Subjects do not learn the distractor, so we do not update
        the weight matrices following it.

        :return:
        """

        # present distractor prior to this list
        self.present_item(self.distractor_idx)

        # if this is a between-list distractor,
        if self.list_idx > 0:
            self.beta_in_play = self.params['beta_rec_post']

        # else if this is the orthogonal item that starts up the system,
        elif self.list_idx == 0:
            self.beta_in_play = 1.0

        # update context regions
        self.update_context_temp()

        # update distractor location for the next list
        self.distractor_idx += 1

        # calculate a vector of primacy gradients (ahead of presenting items)
        prim_vec = (self.params['phi_s'] * np.exp(-self.params['phi_d']
                             * np.asarray(range(self.listlength)))
                      + np.ones(self.listlength))

        # get presentation indices for this particular list:
        thislist_pattern = self.pres_list_nos[self.list_idx]
        thislist_pres_indices = np.searchsorted(
            self.all_session_items_sorted, thislist_pattern)

        # for each item in the current list,
        for i in range(self.listlength):

            # present the item at its appropriate index
            presentation_idx = thislist_pres_indices[i]
            self.present_item(presentation_idx)

            # update the context layer (for now, just the temp region)
            self.beta_in_play = self.params['beta_enc']
            self.update_context_temp()

            # Update the weight matrices

            # Update M_FC
            M_FC_exp = np.dot(self.c_old, self.f_net)
            self.M_FC += np.multiply(M_FC_exp, self.params['gamma_fc'])

            # Update M_CF
            M_CF_exp = np.dot(self.f_net.T, self.c_old.T)
            self.M_CF += np.multiply(M_CF_exp, self.params['gamma_cf']) * prim_vec[i]

            # Update location of study item index
            self.study_item_idx += 1


def separate_files(data_path, subj_id_path):
    """If data is in one big file, separate out the data into sheets, by subject.

    :param data_path: If using this method, data_path should refer directly
        to a single data file containing the consolidated data across all
        subjects.
    :param subj_id_path: path to a list of which subject each list is from.
        lists from a specific subject should all be contiguous and in the
        order in which the lists were presented.
    :return: a list of data matrices, separated out by individual subjects.

    """

    # will contain stimulus matrices presented to each subject
    Ss_data = []

    # for test subject
    data_pres_list_nos = np.loadtxt(data_path, delimiter=',')

    # get list of unique subject IDs

    # use this if dividing a multiple-session subject into sessions
    subj_id_map = np.loadtxt(subj_id_path)
    unique_subj_ids = np.unique(subj_id_map)

    # Get locations where each Subj's data starts & stops.
    new_subj_locs = np.unique(
        np.searchsorted(subj_id_map, subj_id_map))

    # Separate data into sets of lists presented to each subject
    for i in range(new_subj_locs.shape[0]):

        # for all but the last list, get the lists that were presented
        # between the first time that subj ID occurs and when the next ID occurs
        if i < new_subj_locs.shape[0] - 1:
            start_lists = new_subj_locs[i]
            end_lists = new_subj_locs[i + 1]

        # once you have reached the last subj, get all the lists from where
        # that ID first occurs until the final list in the dataset
        else:
            start_lists = new_subj_locs[i]
            end_lists = data_pres_list_nos.shape[0]

        # append subject's sheet
        Ss_data.append(data_pres_list_nos[start_lists:end_lists, :])

    return Ss_data, unique_subj_ids


def run_CMR2_singleSubj(data_mat, LSA_mat, params):

    """Run CMR2 for an individual subject / data sheet"""

    # init. lists to store CMR2 output
    resp_values = []
    RT_values = []
    time_values = []

    # create CMR2 object
    this_CMR = CMR2(
        params=params,
        LSA_mat=LSA_mat, data_mat=data_mat)

    # Present first list
    this_CMR.present_list()

    # layer LSA cos theta values onto the weight matrices
    this_CMR.create_semantic_structure()

    # Recall the first list
    rec_items, RTs_thislist, times_from_start \
        = this_CMR.recall_session()

    # Append values
    resp_values.append(rec_items)
    RT_values.append(RTs_thislist)
    time_values.append(times_from_start)

    # Run CMR2 for all lists after the 0th list
    for i in range(len(this_CMR.pres_list_nos) - 1):
        # present new list
        this_CMR.present_list()

        # recall session
        rec_items_i, RTs_list_i, times_from_start_i \
            = this_CMR.recall_session()

        # append recall responses & times
        resp_values.append(rec_items_i)
        RT_values.append(RTs_list_i)
        time_values.append(times_from_start_i)

    return resp_values, RT_values, time_values


def run_CMR2(LSA_mat, data_path, params, sep_files,
             filename_stem="", source_info_path=".",
             nsource_cells=0, subj_id_path="."):
    """Run CMR2 for all subjects

    time_values = time for each item since beginning of recall session

    For later zero-padding the output, we will get list length from the
    width of presented-items matrix. This assumes equal list lengths
    across Ss and sessions, unless you are inputting each session
    individually as its own matrix, in which case, list length will
    update accordingly.

    If all Subjects' data are combined into one big file, as in some files
    from prior CMR2 papers, then divide data into individual sheets per subj.

    If you want to simulate CMR2 for individual sessions, then you can
    feed in individual session sheets at a time, rather than full subject
    presented-item sheets.
    """

    now_test = time.time()

    # set diagonals of LSA matrix to 0.0
    np.fill_diagonal(LSA_mat, 0)

    # init. lists to store CMR2 output
    resp_vals_allSs = []
    RT_vals_allSs = []
    time_vals_allSs = []

    # Simulate each subject's responses.
    if not sep_files:

        # divide up the data
        subj_presented_data, unique_subj_ids = separate_files(
            data_path, subj_id_path)

        # get list length
        listlength = subj_presented_data[0].shape[1]

        # for each subject's data matrix,
        for m, pres_sheet in enumerate(subj_presented_data):

            subj_id = unique_subj_ids[m]
            print('Subject ID is: ' + str(subj_id))

            resp_Subj, RT_Subj, time_Subj = run_CMR2_singleSubj(
                pres_sheet, LSA_mat,
                params=params)

            resp_vals_allSs.append(resp_Subj)
            RT_vals_allSs.append(RT_Subj)
            time_vals_allSs.append(time_Subj)

    # If files are separate, then read in each file individually
    else:

        # get all the individual data file paths
        indiv_file_paths = glob(data_path + filename_stem + "*.mat")

        # read in the data for each path & stick it in a list of data matrices
        for file_path in indiv_file_paths:

            data_file = scipy.io.loadmat(
                file_path, squeeze_me=True, struct_as_record=False)  # get data
            data_mat = data_file['data'].pres_itemnos  # get presented items

            resp_Subj, RT_Subj, time_Subj = run_CMR2_singleSubj(
                data_mat=data_mat, LSA_mat=LSA_mat,
                params=params)

            resp_vals_allSs.append(resp_Subj)
            RT_vals_allSs.append(RT_Subj)
            time_vals_allSs.append(time_Subj)

        # for later zero-padding the output, get list length from one file.
        data_file = scipy.io.loadmat(indiv_file_paths[0], squeeze_me=True,
                                     struct_as_record=False)
        data_mat = data_file['data'].pres_itemnos

        listlength = data_mat.shape[1]


    ##############
    #
    #   Zero-pad the output
    #
    ##############

    # If more than one subject, reshape the output into a single,
    # consolidated sheet across all Ss
    if len(resp_vals_allSs) > 0:
        resp_values = [item for submat in resp_vals_allSs for item in submat]
        RT_values = [item for submat in RT_vals_allSs for item in submat]
        time_values = [item for submat in time_vals_allSs for item in submat]
    else:
        resp_values = resp_vals_allSs
        RT_values = RT_vals_allSs
        time_values = time_vals_allSs

    # set max width for zero-padded response matrix
    maxlen = listlength * 2

    nlists = len(resp_values)

    # init. zero matrices of desired shape
    resp_mat  = np.zeros((nlists, maxlen))
    RTs_mat   = np.zeros((nlists, maxlen))
    times_mat = np.zeros((nlists, maxlen))

    # place output in from the left
    for row_idx, row in enumerate(resp_values):

        resp_mat[row_idx][:len(row)]  = resp_values[row_idx]
        RTs_mat[row_idx][:len(row)]   = RT_values[row_idx]
        times_mat[row_idx][:len(row)] = time_values[row_idx]

    #print('Analyses complete.')

    print("CMR Time: " + str(time.time() - now_test))

    return resp_mat, times_mat


def main():
    """Main method"""

    # set desired parameters. Example below is for Kahana et al. (2002)

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

    # format printing nicely
    np.set_printoptions(precision=5)

    # set data and LSA matrix paths
    LSA_path = '../K02_files/K02_LSA.txt'
    LSA_mat = np.loadtxt(LSA_path, delimiter=',', dtype=np.float32)

    data_path = '../K02_files/K02_data.txt'
    subjects_path = '../K02_files/K02_subject_ids.txt'

    start_time = time.time()

    # Run the model
    resp, times = run_CMR2(
        LSA_mat=LSA_mat, data_path=data_path,
        params=params_K02, subj_id_path=subjects_path, sep_files=False)

    print("End of time: " + str(time.time() - start_time))

    # save CMR2 results
    np.savetxt('./CMR2_recnos_K02.txt',
               np.asmatrix(resp), delimiter=',', fmt='%i')
    np.savetxt('./CMR2_times_K02.txt.txt',
               np.asmatrix(times), delimiter=',', fmt='%i')


if __name__ == "__main__":
    main()
