def init_q_map():
    return {((x / 10, y / 10, x_sp / 10, y_sp / 10, ang / 10, ang_sp, l, r), a): 0
            for x in range(-10, 12, 2)
            for y in range(-10, 12, 2)
            for x_sp in range(-20, 22, 2)
            for y_sp in range(-20, 22, 2)
            for ang in range(-20, 22, 4)
            for ang_sp in range(-5, 6)
            for l in {0, 1}
            for r in {0, 1}
            for a in range(0, 4)}

# def q_function(q_map, state, action, reward):
#     """
#     Maps a state and an action to a utility. To get the utility function, given
#     a state, simply get the max across all actions. To get the policy, return
#     that action.
#     :param state: the current state
#     :param action: the action to take
#     :param reward: The reward of taking the action given the current state (i.e. the resultant state)
#     :return: the utility for a given <state, action> pair
#     """
#
#     map_key = (tuple(discretize_state(state)), action)
#
#     # Q(s, a) <- (1 - learning_rate) * Q(s, a) + learning_rate(reward + discounting_factor * max{Q(s', a')})
#     learning_rate = 0.5
#     discount_factor = 0.8
#     old_q_value = q_map[map_key]
#
#
#     q_map
#
#     # q_function = {
#     #   (state, action): utility
#     # }
#
#
#     # 1. Initialize all (state,action) pairs to a abitrary fixed value
#     # 2. Iteratively update q_function until convergence (epsilon < ?)
#
#     return q_func