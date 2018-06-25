###################################################
#Example usage:
###################################################
#All files are stored as numpy 'npy'
#arrays. For example, to open RL contact
#map, enter the following code:

mit = np.load('mit_low_res.npy')
lengths = np.load('lengths_low_res.npy')

###################################################
#In order to access the first load the first
#chromosome, enter:

from utility import *
mit_chr20_chr12 = get_contact_map(mit, lengths, chr1=20, chr2=12)

