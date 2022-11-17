import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal


matData = scipy.io.loadmat('Painless021.mat')

print(type(matData))
print(matData)

bp = matData.get('bp')[0] # not sure why you get arrays of arrays
ecg = matData.get('ecg')[0] # note: this is 3x array
ppg = matData.get('ppg')[0]


f_bp = matData.get('f_bp')[0]
f_ecg = matData.get('f_ecg')[0]
f_ppg = matData.get('f_ppg')[0]


plt.figure()
plt.subplot(311)
time_bp = range(0, len(bp)) /f_bp
plt.plot(time_bp, bp)
# plt.ylabel('some numbers')


plt.subplot(312)
time_ecg = range(0, len(ecg)) /f_ecg
plt.plot(time_ecg, ecg)
# plt.ylabel('some numbers')


plt.subplot(313)
time_ppg = range(0, len(ppg)) /f_ppg
plt.plot(time_ppg, ppg)
# plt.ylabel('some numbers')

plt.show(block=False)



# Features
plt.figure()
locSBP, _ = find_peaks(bp, distance=50)
plt.plot(bp)
plt.plot(locSBP, bp[locSBP], "rx")

locDBP, _ = find_peaks(-bp, distance=50)
plt.plot(bp)
plt.plot(locDBP, bp[locDBP], "bx")

plt.show(block=False)


sigLen = np.minimum(len(locSBP), len(locDBP)) #TODO: we really are just dropping data?

tempSBP = bp[locSBP]
SBP = tempSBP[:sigLen]
tempDBP= bp[locDBP][:sigLen]
DBP= tempDBP[:sigLen]
# MBP = (2*DBP + SBP) / 3
MBP = (2*DBP + SBP) / 3


plt.figure()
t_SBP=locSBP[:sigLen] / f_bp;
plt.plot(t_SBP/60,SBP,'r')
# xlabel('beat No')

t_DBP=locDBP[:sigLen] / f_bp;
plt.plot(t_DBP/60, DBP,'b')

t_MBP = range(0, sigLen) #TODO: check this
plt.plot(t_DBP/60, MBP,'g')

# xlabel('time (min)'), ylabel('BP, mmHg')
# legend('SBP', 'DBP')

plt.show(block=False)



# Resample
Fs = np.array([500])

bp_fs = signal.resample(bp, len(bp)*5) # 5 = 500 /100 TODO: fix


ecg_fs = signal.decimate(ecg, 4) #TODO: fix
timeEcg_fs = range(0, len(ecg_fs)) / Fs

ppg_fs = signal.decimate(ppg, 2) #TODO: fix
timePpg_fs = range(0, len(ppg_fs)) /Fs


# Create filters
f_co_hp = 0.5
f_co_lp = 17

# b_highPass = signal.firwin(351, f_co_hp/(Fs/2), pass_zero = 'highpass')
b_highPass, a_highPass = signal.butter(4, (f_co_hp-0.1)/(Fs/2), btype = 'highpass')

w_1, h_1 = signal.freqz(b_highPass, a_highPass, fs=Fs)
fig, ax1 = plt.subplots()

ax1.set_title('Digital high-pass filter frequency response')
ax1.plot(w_1, 20 * np.log10(abs(h_1)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
ax2 = ax1.twinx()

angles = np.unwrap(np.angle(h_1))
ax2.plot(w_1, angles, 'g')
ax2.set_ylabel('Angle (radians)', color='g')
ax2.grid(True)
ax2.axis('tight')
plt.show(block=False)



# b_lowPass = signal.firwin(351, f_co_lp/(Fs/2), pass_zero = 'lowpass')
b_lowPass, a_lowPass = signal.butter(7, f_co_lp/(Fs/2), btype = 'lowpass')

w_2, h_2 = signal.freqz(b_lowPass, a_lowPass, fs=Fs)
fig, ax1 = plt.subplots()

ax1.set_title('Digital low-pass filter frequency response')
ax1.plot(w_2, 20 * np.log10(abs(h_2)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
ax2 = ax1.twinx()

angles = np.unwrap(np.angle(h_2))
ax2.plot(w_2, angles, 'g')
ax2.set_ylabel('Angle (radians)', color='g')
ax2.grid(True)
ax2.axis('tight')
plt.show(block=False)



# Apply filters
ecg_fs = signal.filtfilt(b_highPass, a_highPass, ecg_fs)
ppg_fs = signal.filtfilt(b_highPass, a_highPass, ppg_fs)

ecg_fs = signal.filtfilt(b_lowPass, a_lowPass, ecg_fs)
ppg_fs = signal.filtfilt(b_lowPass, a_lowPass, ppg_fs)

# Output
plt.figure()
plt.subplot(311)
time_bp_fs = range(0, len(bp_fs)) / Fs
plt.plot(time_bp_fs, bp_fs)

plt.subplot(312)
time_ecg_fs = range(0, len(ecg_fs)) / Fs
plt.plot(time_ecg_fs, ecg_fs)

plt.subplot(313)
time_ppg_fs = range(0, len(ppg_fs)) / Fs
plt.plot(time_ppg_fs, ppg_fs)

plt.show(block=False)




#  Again extract the features
plt.figure()
locSBP_f, _ = find_peaks(bp_fs, distance=50*5)
plt.plot(bp_fs)
plt.plot(locSBP_f, bp_fs[locSBP_f], "rx")

locDBP_f, _ = find_peaks(-bp_fs, distance=50*5)
plt.plot(bp_fs)
plt.plot(locDBP_f, bp_fs[locDBP_f], "bx")
plt.show(block=False)


sigLen_fs = np.minimum(len(locSBP_f), len(locDBP_f)) #TODO: we really are just dropping data?

tempSBP_f = bp_fs[locSBP_f]
SBP_f = tempSBP_f[:sigLen_fs]
tempDBP_f= bp_fs[locDBP_f][:sigLen_fs]
DBP_f= tempDBP_f[:sigLen_fs]
MBP_f = (2*DBP_f + SBP_f) / 3


plt.figure()
t_SBP_f=locSBP_f[:sigLen_fs] / Fs;
plt.plot(t_SBP_f/60, SBP_f,'r')
# xlabel('beat No')

t_DBP_f=locDBP_f[:sigLen_fs] / Fs;
plt.plot(t_DBP_f/60, DBP_f,'b')

t_MBP_f = range(0, sigLen_fs) #TODO: check this
plt.plot(t_DBP_f/60, MBP_f,'g')

# xlabel('time (min)'), ylabel('BP, mmHg')
# # legend('SBP', 'DBP')

plt.show(block=False)






















input('Press Enter to exit')