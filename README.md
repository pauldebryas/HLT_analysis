# HLT_analysis

## Main file to run

### run_optimal_HLT.py

- produce json file in "optimal_out/optimal_HLT_mode.json" in which all the most efficient HLT are stored by HNL mass for a given channel mode.

- For each mass, produce the figure of n_passed/n_tot(HLT)

### run_efficiency_HNLmass.py

- Take the "optimal_HLT_mode.json" as input and take the first 5 most common in the list (considering all mass).

- Plot cumulative_efficiency(HNL_mass) and show the contribution of each HLT at each mass point.

## Parameters

### path

local_dir = directory where the HNL data are stored

path_trigger = directory where the triggers lumi file are stored

path_fig = directory where the figures are stored

### cut to apply at Gen level

for tau: cut_tau_pt, cut_tau_eta, cut_tau_id

for electrons: cut_e_pt, cut_e_eta, cut_e_id

for muons: cut_mu_pt, cut_mu_eta, cut_mu_id, cut_mu_iso

### other

not_wanted_HLT = list of triggers that you don't want to consider

color = color for plotting

mode = channel of interest (could be tte, ttm, tee, tmm, tem or all)

trig_types = could be prompt or parking

eps = the smallest, the more HLT will be consider in run_optimal_HLT.py

eff_computation = the number by which you devide the number of events that passed the trigger: could be 

- full: all events in the HNL file
- channel: all the event that passed channel at Gen level
- reco: all events that passed the Reco selection
- channel_reco: ll the event that passed channel and Reco selection 