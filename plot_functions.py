################################################################
################## Showing imbalance? ##########################
################################################################
# fig, ax = plt.subplots(1,1)
# bins = np.arange(-1,3,1)
# ax.set_xlabel('SFH')
# ax.set_ylabel('Number of instances')
# ax.hist(ds3['SFH'], bins=bins, align='left', rwidth=.5)
# ax.set_xticks(bins[:-1])
# plt.show()

# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('SFH')
# ax.set_ylabel('Number of instances')
# ax.hist([ds3['SFH'][ds3['Result'] == 1], ds3['SFH'][ds3['Result'] == -1]], bins=bins, align='left', rwidth=.5)
# ax.set_xticks(bins[:-1])
# plt.show()

# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('popUpWidnow')
# ax.set_ylabel('Number of instances')
# ax.hist(ds3['popUpWidnow'], bins=bins, align='left', rwidth=.5)
# ax.set_xticks(bins[:-1])
# plt.show()

# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('popUpWidnow')
# ax.set_ylabel('Number of instances')
# ax.hist([ds3['popUpWidnow'][ds3['Result'] == 1],ds3['popUpWidnow'][ds3['Result'] == -1]], bins=bins, align='left', rwidth=.5)
# ax.set_xticks(bins[:-1])
# plt.show()
################################################################
#///////////////// Showing imbalance? //////////////////////////
################################################################