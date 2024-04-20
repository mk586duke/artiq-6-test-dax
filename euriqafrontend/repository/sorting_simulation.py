# This file is used to test the complexity (by the required number of swaps) averages all possible initial and final
# configuration of a 23-ion dual-species chain.
# Based on current voltage solutions for splitting and merging, we can't perform merge sort. But for such a highly
# degenarated (since we only have 2 species), we want to estimate a sorting scheme based on bubble sort in
# subunit would be efficient enough.
# Sorting scheme:
# 1. Divide the chain into subunits
#   Count from left, take the shortest part with element sum to zero of the error array (current configuration -
#   desired configuration, 171:0, 172:1) as a subunit (for parts consists only of zeros, for each part we define a
#   single trivial subunit).
#   By far, the problem has been reduced to subspaces in which we don't do swap of ions between subspaces, also we
#   don't need to care about trivial subspaces.
# 2. Swap all adjacent pair with no zero in between in all subunits (corresponding to -1+1 or +1-1 pair in the error
#   array with no 0 in between)
# 3. Subunit sort:
#   Central problem: reduce the subunit length (it converges)
#   Principle:
#   (1) Fixing the subunit only involves swapping the ions within the subunit (the property of subunit)
#   (2) Swapping an ion pair with one errored ion doesn't increase the number of errors (total number of errors is
#   conserved)
#   (3) Strict swap of distant pair via nearest neighboring swap is expensive (due to the moving back swap in order
#   to keep the ions in between the distant pair intact)
#
#    (1) Find the error closest to the rightmost error but with different sign, move it to the rightmost position by
#   nearest neighboring swap to eliminate the error. After movement the subunit length must be reduced at least by one
#   (since at least the rightmost error is eliminated, and the leftmost element and its neighbor in this subunit either
#   have their error array stay the same, or become 0, +-1 with the error position shifted by one, thus shrink the
#   subunit by one more, and the error sign changed).
#   (2) Perform 1, 2
#
#   Other sorting schemes:
#   Bubble sort: stuck, didn't think further about how to fix it, and bubble sort seems to waste many swaps on swapping
#   ion pair with the same error.
#   Balance-based sort

# DC: Desired Configuration;
# CC: Current Configuration
# 1: 172; 0: 171;
# Error: CC - DC
# flag: True when CC == DC
# swapnum: number of swaps, updated in time

import numpy as np
import random
import matplotlib.pyplot as plt


# def update_error(DC, CC):
#     return [CC[i] - DC[i] for i in range(len(CC))]
#
#
# def find_subunit(DC, CC):
#     Error = update_error(DC, CC)
#     # print("Error")
#     # print(Error)
#     # localnode: the python index of the last element in the subunit
#     trivial_startnode = []
#     trivial_endnode = []
#     startnode = []
#     endnode = []
#     localsum = 0
#     localnode = 0
#     trivialflag = True
#
#     for ind in range(len(DC)):
#         localsum += Error[ind]
#
#         # trivial subunit starts (as the first subunit)
#         if ind == 0 and Error[ind] == 0:
#             trivial_startnode.append(localnode)
#
#         # trivial subunit ends, nontrivial subunit starts
#         if localsum != 0 and trivialflag == True:
#             trivial_endnode.append(ind - 1)
#             localnode = ind
#             startnode.append(localnode)
#             trivialflag = False
#
#         # nontrivial subunit ends, trivial/nontrivial subunit starts
#         if localsum == 0 and trivialflag == False:
#             endnode.append(ind)
#             localnode = ind + 1
#             localsum = 0
#             if ind < len(DC) - 1:
#                 if Error[ind + 1] == 0:
#                     trivialflag = True
#                     trivial_startnode.append(localnode)
#                 else:
#                     trivialflag = False
#                     startnode.append(localnode)
#
#         # trivial subunit ends (as the last subunit)
#         if ind == len(DC) - 1 and Error[ind] == 0:
#             trivial_endnode.append(ind)
#
#     return trivial_startnode, trivial_endnode, startnode, endnode
#
# def pair_eliminator(leftind, rightind, CC, swapnum):
#     CC[leftind], CC[rightind] = CC[rightind], CC[leftind]
#     swapnum += 1
#     return CC
#
#
# def subunit_pair_eliminator(DC, CC, startnode, endnode, swapnum):
#     # print("Error before eliminator")
#     # print(update_error(DC, CC))
#     # current configuration of the subunit before moving the leftmost ion to the rightmost opposite error position
#     sub_config = CC[startnode:endnode + 1]
#     # current error array of the subunit
#     sub_error = [sub_config[i] - DC[i + startnode] for i in range(len(sub_config))]
#     # swap the ion pair with opposite errors
#     for i in range(len(sub_error) - 1):
#         if sub_error[i] * sub_error[i + 1] == -1:
#             sub_config[i], sub_config[i + 1] = sub_config[i + 1], sub_config[i]
#             sub_error[i] = 0
#             sub_error[i + 1] = 0
#             swapnum += 1
#             # print("Pair elimination")
#             # print(swapnum)
#
#     # update CC
#     new_CC = CC[:startnode]
#     new_CC.extend(sub_config)
#     new_CC.extend(CC[endnode + 1:])
#     # print("Oops")
#     # print(CC)
#     # print(new_CC)
#     return new_CC, swapnum
#
#
# def subunit_sort_movingstep(DC, CC, startnode, endnode, swapnum):
#     # current configuration of the subunit before moving the leftmost ion to the rightmost opposite error position
#     sub_config = CC[startnode:endnode + 1]
#     # current error array of the subunit
#     sub_error = [sub_config[i] - DC[i + startnode] for i in range(len(sub_config))]
#     # sign of the target error is opposite to the sign of the leftmost error
#     target_error = - sub_error[0]
#     # find the index of target opposite error that is closest to the leftmost error
#     for i in range(len(sub_error)):
#         if sub_error[i] == target_error:
#             target_index = i
#             break
#     # update the new subunit configuration after moving (move the ion at the leftmost to after the target ion)
#     new_sub_config = sub_config[1:target_index + 1]
#     new_sub_config.append(sub_config[0])
#     new_sub_config.extend(sub_config[target_index + 1:])
#     # prepare the total current configuration after moving
#     new_CC = CC[:startnode]
#     new_CC.extend(new_sub_config)
#     new_CC.extend(CC[endnode + 1:])
#     # update the swapnum by adding the swaps required for moving
#     swapnum += target_index
#     # print("moving")
#     # print(swapnum)
#     # print("Error")
#     # print(update_error(DC, CC))
#     # print("old")
#     # print(CC)
#     # print("moved")
#     # print(new_CC)
#     # print(update_error(DC, new_CC))
#     return new_CC, swapnum
#
#
# def subunit_sort(DC, CC, startnode, endnode, swapnum):
#     # if need to do it "vertically" we need this function
#     # current configuration of the subunit before moving the leftmost ion to the rightmost opposite error position
#     sub_config = CC[startnode:endnode + 1]
#     # corresponding DC of the subunit
#     dc_sub_config = DC[startnode:endnode + 1]
#     # note: if needs to be translated to the whole chain from subunit, need to add startnode
#     sub_trivial_startnode, sub_trivial_endnode, sub_startnode, sub_endnode = find_subunit(dc_sub_config, sub_config)
#     sub_startnode_off = 0
#     # by moving sort and pair elimination to completely sort this subunit
#     while len(sub_startnode) != 0:
#         CC, swapnum = subunit_sort_movingstep(DC, CC, startnode + sub_startnode_off, endnode, swapnum)
#         CC, swapnum = subunit_pair_eliminator(DC, CC, startnode + sub_startnode_off, endnode, swapnum)
#         sub_config = CC[startnode:endnode + 1]
#         sub_trivial_startnode, sub_trivial_endnode, sub_startnode, sub_endnode = find_subunit(dc_sub_config, sub_config)
#         if len(sub_startnode) > 0:
#             sub_startnode_off = sub_startnode[0]
#
#     return CC, swapnum
#
#
# def vertical_sort(DC, CC):
#     # Vertically sort
#     # print(CC)
#     # print(DC)
#     # print(update_error(DC, CC))
#     trivial_startnode, trivial_endnode, startnode, endnode = find_subunit(DC, CC)
#     swapnum = 0
#     while len(startnode) != 0:
#         # check vertical sort validity
#         # for each round go over all the subunits decided during the previous round, do the moving swap, then eliminate
#         # pairs, then calculate new subunits.
#         temp_startnode = startnode
#         temp_endnode = endnode
#         for i in range(len(temp_startnode)):
#             CC, swapnum = subunit_sort(DC, CC, temp_startnode[i], temp_endnode[i], swapnum)
#
#         trivial_startnode, trivial_endnode, startnode, endnode = find_subunit(DC, CC)
#
#     # print("Done!")
#     # print(swapnum)
#     # print(CC)
#     # print(DC)
#
#     return swapnum
#
#
# def vertical_sort_stat_estimate(Nions, Ncool):
#     shotnum = 100000
#     shot = 1
#     swapnum_list = []
#     while shot <= shotnum:
#         # randomly generate the initial and final config
#         # DC 172 ions' indices
#         DCcool = []
#         DCcoolind = random.randint(0, Nions - 1)
#         while len(DCcool) < Ncool:
#             if DCcoolind not in DCcool:
#                 DCcool.append(DCcoolind)
#                 DCcoolind = random.randint(0, Nions - 1)
#             else:
#                 DCcoolind = random.randint(0, Nions - 1)
#         # DC
#         DC = []
#         for i in range(Nions):
#             if i in DCcool:
#                 DC.append(1)
#             else:
#                 DC.append(0)
#         # CC 172 ions' indices
#         CCcool = []
#         CCcoolind = random.randint(0, Nions - 1)
#         while len(CCcool) < Ncool:
#             if CCcoolind not in CCcool:
#                 CCcool.append(CCcoolind)
#                 CCcoolind = random.randint(0, Nions - 1)
#             else:
#                 CCcoolind = random.randint(0, Nions - 1)
#         # CC
#         CC = []
#         for i in range(Nions):
#             if i in CCcool:
#                 CC.append(1)
#             else:
#                 CC.append(0)
#
#         swapnum = vertical_sort(DC, CC)
#         swapnum_list.append(swapnum)
#
#         shot += 1
#
#     return (sum(swapnum_list) / shotnum)
#
#
# def horizontal_sort(DC, CC):
#     # Horizontally sort
#     # TODO: to confirm in the step of moving, it may change the global division of subunits?
#     # Debugged!
#     # print(DC)
#     # print(CC)
#     trivial_startnode, trivial_endnode, startnode, endnode = find_subunit(DC, CC)
#     swapnum = 0
#     while len(startnode) != 0:
#         # for each round go over all the subunits decided during the previous round, do the moving swap, then eliminate
#         # pairs, then calculate new subunits.
#         temp_startnode = startnode
#         temp_endnode = endnode
#         for i in range(len(temp_startnode)):
#             CC, swapnum = subunit_sort_movingstep(DC, CC, temp_startnode[i], temp_endnode[i], swapnum)
#
#             CC, swapnum = subunit_pair_eliminator(DC, CC, temp_startnode[i], temp_endnode[i], swapnum)
#
#         trivial_startnode, trivial_endnode, startnode, endnode = find_subunit(DC, CC)
#     # print("Done!")
#     # print(swapnum)
#     # print(CC)
#     # print(DC)
#     return swapnum
#
#
# def horizontal_sort_stat_estimate(Nions, Ncool):
#     shotnum = 100000
#     shot = 1
#     swapnum_list = []
#     while shot <= shotnum:
#         # randomly generate the initial and final config
#         # DC 172 ions' indices
#         DCcool = []
#         DCcoolind = random.randint(0, Nions - 1)
#         while len(DCcool) < Ncool:
#             if DCcoolind not in DCcool:
#                 DCcool.append(DCcoolind)
#                 DCcoolind = random.randint(0, Nions - 1)
#             else:
#                 DCcoolind = random.randint(0, Nions - 1)
#         # DC
#         DC = []
#         for i in range(Nions):
#             if i in DCcool:
#                 DC.append(1)
#             else:
#                 DC.append(0)
#         # CC 172 ions' indices
#         CCcool = []
#         CCcoolind = random.randint(0, Nions - 1)
#         while len(CCcool) < Ncool:
#             if CCcoolind not in CCcool:
#                 CCcool.append(CCcoolind)
#                 CCcoolind = random.randint(0, Nions - 1)
#             else:
#                 CCcoolind = random.randint(0, Nions - 1)
#         # CC
#         CC = []
#         for i in range(Nions):
#             if i in CCcool:
#                 CC.append(1)
#             else:
#                 CC.append(0)
#
#         swapnum = horizontal_sort(DC, CC)
#         swapnum_list.append(swapnum)
#
#         shot += 1
#
#     return (sum(swapnum_list) / shotnum)
#
#
# # balance sort
# # Divide the chain in half (middle ion belongs to left part), move the 172 closest to the division line from unbalanced side to
# # the other side, at the same time move the 171 from the other side closest to the middle line to the other side, then subdivide
# # the half chain into half again, until it converges to single ion
#
# def balance_calculator(DC, CC, startnode, endnode):
#     # Checked!
#     # under the scheme of balance sort, we always only need to check balance of one side in one division, if that side is
#     # balanced, the other side must be balanced
#     # Positive balance: more 172 in CC
#     if sum(CC[startnode:endnode + 1]) > sum(DC[startnode:endnode + 1]):
#         balance = 1
#     elif sum(CC[startnode:endnode + 1]) < sum(DC[startnode:endnode + 1]):
#         balance = -1
#     else:
#         balance = 0
#     return balance
#
# def divider_and_balancer(DC, CC, startnode, endnode, startnode_list, part_config, endnode_list, error_list, swapnum):
#     # don't use the global variables instead of creating a new one and return them in case messing up due to unexpected mistakes
#     # startnode and endnode are both 0 indexed
#     leftstartnode = startnode
#     rightendnode = endnode
#     # balance = +1, left unbalanced; -1, right unbalanced; 0: balanced
#     if startnode == endnode:
#         # print("Only one ion left!")
#         new_part_config = part_config
#         new_error_list = error_list
#         startnode_list = startnode_list
#         endnode_list = endnode_list
#     else:
#         if endnode % 2 == 0:
#             leftendnode = int((startnode + endnode - 1) / 2)
#             rightstartnode = int((startnode + endnode + 1) / 2)
#             balance = balance_calculator(DC, CC, leftstartnode, leftendnode)
#         elif endnode % 2 == 1:
#             leftendnode = int((startnode + endnode) / 2)
#             rightstartnode = int((startnode + endnode) / 2) + 1
#             balance = balance_calculator(DC, CC, leftstartnode, leftendnode)
#         # update the startnode and endnode list
#         startnode_list.insert(startnode_list.index(startnode) + 1, rightstartnode)
#         endnode_list.insert(endnode_list.index(endnode), leftendnode)
#         # balancing by swapping
#         while balance != 0:
#             if balance == -1:
#                 # find the closest 172 on the right, get the index of it before moving to the left
#                 for i in range(rightstartnode, rightendnode + 1):
#                     if CC[i] == 1:
#                         moveind_coolant = i
#                         break
#                         # find the closest 171 on the left, get the index of it before moving to the right
#                 for i in range(leftendnode, leftstartnode - 1, -1):
#                     if CC[i] == 0:
#                         moveind_qubit = i
#                         break
#                 # move the 171 right to right next to the division line
#                 for i in range(1, leftendnode - moveind_qubit + 1):
#                     CC[moveind_qubit + i - 1], CC[moveind_qubit + i] = CC[moveind_qubit + i], CC[
#                         moveind_qubit + i - 1]
#                     swapnum += 1
#                 # move the 172 left to right next to the division line
#                 for i in range(1, moveind_coolant - rightstartnode + 1):
#                     CC[moveind_coolant - i], CC[moveind_coolant - i + 1] = CC[moveind_coolant - i + 1], CC[
#                         moveind_coolant - i]
#                     swapnum += 1
#
#             elif balance == 1:
#                 # find the closest 171 on the right, get the index of it before moving to the left
#                 for i in range(rightstartnode, rightendnode + 1):
#                     if CC[i] == 0:
#                         moveind_qubit = i
#                         break
#                 # find the closest 172 on the left, get the index of it before moving to the right
#                 for i in range(leftendnode, leftstartnode - 1, -1):
#                     if CC[i] == 1:
#                         moveind_coolant = i
#                         break
#                 # move the 172 right to right next to the division line
#                 for i in range(1, leftendnode - moveind_coolant + 1):
#                     CC[moveind_coolant + i - 1], CC[moveind_coolant + i] = CC[moveind_coolant + i], CC[
#                         moveind_coolant + i - 1]
#                     swapnum += 1
#                 # move the 171 left to right next to the division line
#                 for i in range(1, moveind_qubit - rightstartnode + 1):
#                     CC[moveind_qubit - i], CC[moveind_qubit - i + 1] = CC[moveind_qubit - i + 1], CC[
#                         moveind_qubit - i]
#                     swapnum += 1
#
#             # swap the 171 and 172
#             CC[leftendnode], CC[rightstartnode] = CC[rightstartnode], CC[leftendnode]
#             swapnum += 1
#
#             # update balance
#             balance = balance_calculator(DC, CC, leftstartnode, leftendnode)
#
#         new_error_list = update_error(DC, CC)
#
#         # update the part_config
#         new_part_config = []
#         for i in range(startnode_list.index(startnode)):
#             new_part_config.append(part_config[i])
#         new_part_config.append(CC[leftstartnode:leftendnode + 1])
#         new_part_config.append(CC[rightstartnode:rightendnode + 1])
#         for i in range(startnode_list.index(startnode) + 1, len(part_config), 1):
#             new_part_config.append(part_config[i])
#
#     return CC, new_part_config, new_error_list, startnode_list, endnode_list, swapnum
#
# def binder(part_config):
#     CC = []
#     for i in range(len(part_config)):
#         CC.extend(part_config[i])
#     return CC
#
#
# def error_flag(Error):
#     # used both for calculating the error flag and balance flag
#     error_flag = 0
#     for i in Error:
#         error_flag += abs(i)
#     return error_flag
#
#
# def balance_sort(DC, CC):
#     # configuration of each part arranged sequentially[[part1], ..., [partm]]
#     part_config = []
#     part_config.append(CC)
#     # startnode of each part
#     startnode_list = [0]
#     endnode_list = [len(DC) - 1]
#
#     error_list = update_error(DC, CC)
#     Error_flag = error_flag(error_list)
#     swapnum = 0
#
#     while Error_flag != 0:
#         for i in range(len(startnode_list)):
#             # get the startnode and endnode of the part to be divided
#             startnode = startnode_list[i]
#             endnode = endnode_list[i]
#             # divide, balance, and update CC, balance_list, part_config, startnode_list, endnode_list
#             CC, part_config, error_list, startnode_list, endnode_list, swapnum = \
#                 divider_and_balancer(DC, CC, startnode, endnode, startnode_list, part_config, endnode_list, error_list, swapnum)
#         Error_flag = error_flag(error_list)
#
#     # print("DC", DC)
#     # print("CC", CC)
#     # print("Swap number", swapnum)
#
#     return swapnum
#
# def balance_sort_stat_estimate(Nions, Ncool):
#     shotnum = 10000
#     shot = 1
#     swapnum_list = []
#     swapnum_max = 0
#     while shot <= shotnum:
#         # randomly generate the initial and final config
#         # DC 172 ions' indices
#         DCcool = []
#         DCcoolind = random.randint(0, Nions - 1)
#         while len(DCcool) < Ncool:
#             if DCcoolind not in DCcool:
#                 DCcool.append(DCcoolind)
#                 DCcoolind = random.randint(0, Nions - 1)
#             else:
#                 DCcoolind = random.randint(0, Nions - 1)
#         # DC
#         DC = []
#         for i in range(Nions):
#             if i in DCcool:
#                 DC.append(1)
#             else:
#                 DC.append(0)
#         # CC 172 ions' indices
#         CCcool = []
#         CCcoolind = random.randint(0, Nions - 1)
#         while len(CCcool) < Ncool:
#             if CCcoolind not in CCcool:
#                 CCcool.append(CCcoolind)
#                 CCcoolind = random.randint(0, Nions - 1)
#             else:
#                 CCcoolind = random.randint(0, Nions - 1)
#         # CC
#         CC = []
#         for i in range(Nions):
#             if i in CCcool:
#                 CC.append(1)
#             else:
#                 CC.append(0)
#
#         swapnum = balance_sort(DC, CC)
#         swapnum_list.append(swapnum)
#
#         if swapnum > swapnum_max:
#             swapnum_max = swapnum
#
#         shot += 1
#
#     return (sum(swapnum_list) / shotnum), swapnum_max
#
# # Optimal case corresponds to the no crossing connection between 172 ions in CC and DC
# def no_crossing_sort(DC, CC):
#     DC_coolant_index = []
#     CC_coolant_index = []
#     swapnum = 0
#     # solution: k, means swap (k, k + 1) pair
#     swapsol = []
#
#     for i in range(len(DC)):
#         if DC[i] == 1:
#             DC_coolant_index.append(i)
#         if CC[i] == 1:
#             CC_coolant_index.append(i)
#
#     # Left movement starts from left; right movement starts from right
#     # Bookkeeping the direction of each 172 ion
#     # > 0: moving left ; <0: moving right.
#     # movement in two direction is completely decoupled
#     Dir = [CC_coolant_index[i] - DC_coolant_index[i] for i in range(len(CC_coolant_index))]
#
#     for i in range(len(Dir)):
#         if Dir[i] > 0:
#             CC[CC_coolant_index[i]], CC[DC_coolant_index[i]] = CC[DC_coolant_index[i]], CC[CC_coolant_index[i]]
#             swapnum += abs(CC_coolant_index[i] - DC_coolant_index[i])
#             for k in range(CC_coolant_index[i] - 1, DC_coolant_index[i] - 1, -1):
#                 swapsol.append(k)

#     for i in range(len(Dir) - 1, -1, -1):
#         if Dir[i] < 0:
#             CC[CC_coolant_index[i]], CC[DC_coolant_index[i]] = CC[DC_coolant_index[i]], CC[CC_coolant_index[i]]
#             swapnum += abs(CC_coolant_index[i] - DC_coolant_index[i])
#             for k in range(CC_coolant_index[i], DC_coolant_index[i]):
#                 swapsol.append(k)
#
#     return swapnum, swapsol
#
# def no_crossing_sort_stat_estimate(Nions, Ncool):
#     shotnum = 10000
#     swapnum_max = 0
#     shot = 1
#     swapnum_list = []
#     while shot <= shotnum:
#         # randomly generate the initial and final config
#         # DC 172 ions' indices
#         DCcool = []
#         DCcoolind = random.randint(0, Nions - 1)
#         while len(DCcool) < Ncool:
#             if DCcoolind not in DCcool:
#                 DCcool.append(DCcoolind)
#                 DCcoolind = random.randint(0, Nions - 1)
#             else:
#                 DCcoolind = random.randint(0, Nions - 1)
#         # DC
#         DC = []
#         for i in range(Nions):
#             if i in DCcool:
#                 DC.append(1)
#             else:
#                 DC.append(0)
#
#         # DC = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
#
#         # CC 172 ions' indices
#         CCcool = []
#         CCcoolind = random.randint(0, Nions - 1)
#         while len(CCcool) < Ncool:
#             if CCcoolind not in CCcool:
#                 CCcool.append(CCcoolind)
#                 CCcoolind = random.randint(0, Nions - 1)
#             else:
#                 CCcoolind = random.randint(0, Nions - 1)
#         # CC
#         CC = []
#         for i in range(Nions):
#             if i in CCcool:
#                 CC.append(1)
#             else:
#                 CC.append(0)
#
#         swapnum, swapsol = no_crossing_sort(DC, CC)
#         swapnum_list.append(swapnum)
#
#         if swapnum > swapnum_max:
#             swapnum_max = swapnum
#
#         shot += 1
#
#     return (sum(swapnum_list) / shotnum, swapnum_max)
#
# #DC = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# #CC = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#
# # swapnum, swapnum_max = no_crossing_sort_stat_estimate(23, 5)
# # print(swapnum)
# # print(swapnum_max)
#
# Result = []
# Upperlimit = []
# Result_balance = []
# Upperlimit_balance = []
# # Nion = np.linspace(10, 100, 91)
# # for i in range(10, 101):
# #     swapnum, swapnum_max = no_crossing_sort_stat_estimate(i, 5)
# #     Result.append(swapnum)
# #     Upperlimit.append(swapnum_max)
# #     swapnum_balance, swapnum_max_balance = balance_sort_stat_estimate(i, 5)
# #     Result_balance.append(swapnum_balance)
# #     Upperlimit_balance.append(swapnum_max_balance)
#
# # plt.plot(Nion, Result, label="Average (no crossing)")
# # plt.plot(Nion, Upperlimit, label="Max (no crossing)")
# # plt.plot(Nion, Result_balance, label="Average (balance)")
# # plt.plot(Nion, Upperlimit_balance, label="Max (balance)")
# # plt.xlabel("N")
# # plt.ylabel("Number of swap")
# # plt.title("N-ion chain with 5 coolants, arbitrary initial and final configuration")
# # plt.legend()
# # plt.show()
#
# Ncool = np.linspace(1, 22, 22)
# for i in range(1, 23):
#     swapnum, swapnum_max = no_crossing_sort_stat_estimate(23, i)
#     swapnum_balance, swapnum_max_balance = balance_sort_stat_estimate(23, i)
#     Result.append(swapnum)
#     Upperlimit.append(swapnum_max)
#     Result_balance.append(swapnum_balance)
#     Upperlimit_balance.append(swapnum_max_balance)
#
# plt.plot(Ncool, Result, label="Average (no crossing)")
# plt.plot(Ncool, Upperlimit, label="Max (no crossing)")
# plt.plot(Ncool, Result_balance, label="Average (balance)")
# plt.plot(Ncool, Upperlimit_balance, label="Max (balance)")
# plt.xlabel("Ncool")
# plt.ylabel("Number of swap")
# plt.title("23-ion chain with Ncool coolants, arbitrary initial and final configuration")
# plt.legend()
# plt.show()

