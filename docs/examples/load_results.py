import numpy as np
import matplotlib.pyplot as plt

result_dict = np.load("result_dict.npy", allow_pickle=True)
chirp_transform_delta_B = result_dict.item()['chirp_transform_delta_B']
chirp_transform_delta_f = result_dict.item()['chirp_transform_delta_f']
B_diff = result_dict.item()['B_diff']
f_diff = result_dict.item()['f_diff']
B_diff_error_rx0 = (100*(chirp_transform_delta_B-B_diff[:,:,None])/B_diff[:,:,None])[:,0,:]
f_diff_error_rx0 = (100*(chirp_transform_delta_f-f_diff[:,:,None])/f_diff[:,:,None])[:,0,:]
fig, (ax_f, ax_B) = plt.subplots(2,1,num="error_plot")
ax_f.plot(f_diff_error_rx0.T)
ax_f.legend(["tx0", "tx1", "tx2", "tx3"])
ax_f.set_xlabel("chirp idx")
ax_f.set_ylabel("f Estimation error in %")
ax_f.set_xlim([0, f_diff_error_rx0.shape[-1]])
ax_f.grid()
ax_B.plot(B_diff_error_rx0.T)
ax_B.legend(["tx0", "tx1", "tx2", "tx3"])
ax_B.set_xlabel("chirp idx")
ax_B.set_ylabel("B Estimation error in %")
ax_B.set_xlim([0, B_diff_error_rx0.shape[-1]])
ax_B.grid()

plt.figure()
plt.hist(f_diff_error_rx0[0])
plt.figure()
plt.hist(B_diff_error_rx0[0])
plt.show()
