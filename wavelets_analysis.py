# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2023/10/16 22:23:13 
 
@author: BUUJUN WANG
"""

#%%
def WaveTrans(series, dt, **kwards_cwt):
    import pycwt as cwt
    import numpy as np

    series = np.asarray(series)
    std = series.std()
    var = series.var()
    mean = series.mean()
    N = len(series)
    series_norm = (series-mean)/std

    mother = cwt.Morlet(6)
    alpha, _, _ = cwt.ar1(series)  # Lag-1 autocorrelation for red noise

    wave, scales, freqs, coi, fft, fft_freqs = \
        cwt.cwt(series_norm, dt=dt, wavelet=mother, **kwards_cwt)
    power = np.abs(wave)**2 #/ scales[:, None]
    periods = 1./freqs
    fft_power = (np.abs(fft)**2) * var
    fft_periods = 1./fft_freqs

    signif, fft_theor = cwt.significance(1.0, dt=dt, 
                                         scales=scales,
                                         sigma_test=0, 
                                         alpha=alpha, 
                                         significance_level=0.95,  
                                         wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = np.abs(wave)**2 / sig95 # The power is significant where the ratio power / sig95 > 1.
    fft_theor = fft_theor * var

    glbl_power = power.mean(axis=1) * var
    dof = N - scales  # Correction for padding at edges
    glbl_sig95, glbl_fft_theor = \
        cwt.significance(var, dt=dt, scales=scales, 
                         sigma_test=1, alpha=alpha,
                         significance_level=0.95, 
                         dof=dof, wavelet=mother)
    
    iwave = (np.asarray(cwt.icwt(wave, scales, dt, wavelet=mother), float) * std) + mean

    result_dict = dict(iwave=iwave, power=power, 
                       scales=scales, periods=periods, 
                       coi=coi, sig95=sig95, 
                       glbl_power=glbl_power, glbl_sig95=glbl_sig95,
                       fft_power=fft_power, fft_periods=fft_periods,
                       fft_theor=fft_theor)

    return result_dict

def Wave_Analysis_Plot(dat, t, dt, log=False):
    import proplot as pplt
    import numpy as np

    res_wt = WaveTrans(dat, dt)
    iwave = res_wt['iwave']
    periods = res_wt['periods']
    power = res_wt['power']
    sig95 = res_wt['sig95']
    coi = res_wt['coi']
    glbl_power = res_wt['glbl_power']
    glbl_sig95 = res_wt['glbl_sig95']
    fft_power = res_wt['fft_power']
    fft_periods = res_wt['fft_periods']
    fft_theor = res_wt['fft_theor']

    fig, axes = pplt.subplots([[1, 1, 1, 1, 0], 
                               [2, 2, 2, 2, 3]], figsize=(12, 10))
    axes[0].plot(t, iwave, c='grey', lw=1.5)
    axes[0].plot(t, dat, c='k', lw=2.5)
    if log==True:   axes[1].contourf(t, periods, np.log2(power), cmap='Fire')
    else:           axes[1].contourf(t, periods, power, cmap='Fire')
    axes[1].contour(t, periods, sig95, colors='k', vmin=1, vmax=1)
    axes[1].plot(t, coi, c='k', lw=3)

    axes[2].plot(fft_power, fft_periods, c='grey', lw=2)
    axes[2].plot(fft_theor, periods, c='grey', ls='--', lw=2)
    axes[2].plot(glbl_power, periods, c='k', lw=2)
    axes[2].plot(glbl_sig95, periods, c='k', ls='--', lw=2)

    axes[0].format(ylabel='Data')
    axes[2].format(xlabel='Power', xlim=(0, fft_power[fft_periods<=coi.max()].max()))
    axes[0:2].format(xlabel='Time')
    axes[1:3].format(ylabel='Periods', yscale='log', 
                    yticks=2**np.arange(1, int(np.log2(coi.max()))), 
                    ylim=(coi.max()*1.1, periods[0]))                 
    pplt.show()

    return res_wt


# #%% 
# # TEST 1
# dt = 1
# year = np.arange(1900, 2000, dt)
# dat = np.concatenate(
#     [np.sin(np.arange(int(year.size/3))*np.pi/2), 
#      np.sin(np.arange(int(year.size/3), int(year.size/3*2))*np.pi/4),
#      np.sin(np.arange(int(year.size/3*2), int(year.size))*np.pi/2)])

# res_wt = Wave_Analysis_Plot(dat, year, dt)

# print('')

# #%% 
# # TEST 2
# url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
# dat = np.genfromtxt(url, skip_header=19)
# t0 = 1871.0
# dt = 0.25  # In years
# N = dat.size
# t = np.arange(0, N) * dt + t0

# Wave_Analysis_Plot(dat, t, dt)

# print('')

# # %%