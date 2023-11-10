import numpy as np
import os
import csv


phy_dirs = [
    '/Users/leo/Downloads/phy_folder/poke2_230525_123745_0-34seconds',
    '/Users/leo/Downloads/phy_folder/poke2_230525_123745_34-204seconds',
    '/Users/leo/Downloads/phy_folder/poke2_230525_123745_204-endseconds'
]
dir_start_times = [0, 34, 204]

new_dir = '/Volumes/PikesPeak/VNCMP/2023-05-25/phy_folder/poke2_230525_123745'

# From each dir read files of interest, load, change timestamps, append to new files
max_clust_name = 0
new_cluster_group = []
for i, dir in enumerate(phy_dirs):
    spike_times = np.load(os.path.join(dir, 'spike_times.npy'))
    spike_clusters = np.load(os.path.join(dir, 'spike_clusters.npy'))
    # Modify
    spike_times += dir_start_times[i] * 30000
    clust_ids = np.unique(spike_clusters)
    id_dict = {id : id + max_clust_name + 1 for id in clust_ids}
    for id in clust_ids:
        spike_clusters[spike_clusters == id] = id_dict[id]
    with open(os.path.join(dir, 'cluster_group.tsv')) as file:
        cluster_group = []
        tsv_file = csv.reader(file, delimiter='\t')
        next(tsv_file, None)
        for line in tsv_file:
            cluster_group.append([str(id_dict[int(line[0])]), line[1]])
    # Save 
    if i == 0:
        new_spike_times = spike_times
        new_spike_clusters = spike_clusters
        new_cluster_group = [['cluster_id', 'group'], *cluster_group]
    else:
        new_spike_times = np.vstack((new_spike_times, spike_times))
        new_spike_clusters = np.vstack((new_spike_clusters, np.reshape(spike_clusters, (-1, 1))))
        new_cluster_group += cluster_group
    max_clust_name = np.max(np.unique(spike_clusters))

np.save(os.path.join(new_dir, 'spike_times.npy'), new_spike_times)
np.save(os.path.join(new_dir, 'spike_clusters.npy'), new_spike_clusters)
np.savetxt(os.path.join(new_dir, 'cluster_group.tsv'), new_cluster_group, delimiter='\t', fmt='%s')