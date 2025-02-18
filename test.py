from visualize_b0map import view_B0map
import scipy


matlab_data = scipy.io.loadmat('B0/_1/B0map_echos_segue_masked.mat')
B0map_matlab = matlab_data['B0map']
view_B0map(B0map_matlab)
