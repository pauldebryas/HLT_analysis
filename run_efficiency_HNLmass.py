#all import

import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import statsmodels.api as sm 

from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents import NanoAODSchema
from coffea import processor, hist
from statsmodels.stats.proportion import proportion_confint 
NanoAODSchema.warn_missing_crossrefs = True # no strict cross-references gives warning !

from raw_samples import raw_signal_samples
from helpers import files_from_dir
from tqdm import tqdm
import json
import pandas as pd

from utils import load_triggers,lepton_mask_gen, reco_mask_cut, plot_cumulative_eff_mass, plot_raw_eff_mass, cumu_eff,raw_eff

# custom HNLnanoAODschema class: for now, our custom NanoAOD do not have a strict cross-references convention as standard NanoAOD, so we make an exception for our custom Branches (in 'excluded_list')

import warnings
from coffea.nanoevents import transforms
from coffea.nanoevents.util import quote, concat
from coffea.nanoevents.schemas.base import BaseSchema, listarray_form, zip_forms, nest_jagged_forms


from coffea.nanoevents import transforms
from coffea.nanoevents.schemas import NanoAODSchema
class HNLnanoAODschema(NanoAODSchema):
    def _build_collections(self, branch_forms):
        # parse into high-level records (collections, list collections, and singletons)
        collections = set(k.split("_")[0] for k in branch_forms)
        collections -= set(
            k for k in collections if k.startswith("n") and k[1:] in collections
        )

        # Create offsets virtual arrays
        for name in collections:
            if "n" + name in branch_forms:
                branch_forms["o" + name] = transforms.counts2offsets_form(
                    branch_forms["n" + name]
                )

        # Create global index virtual arrays for indirection
        excluded_list= {'PatDSAMuon_l1Idx','PatDSAMuon_l2Idx','DiMuon_l1Idx','DiMuon_l2Idx','DiDSAMuon_l1Idx','DiDSAMuon_l2Idx'}
        idxbranches = [k for k in branch_forms if "Idx" in k and k not in excluded_list]
        for name in collections:
            indexers = [k for k in idxbranches if k.startswith(name + "_")]
            for k in indexers:
                target = k[len(name) + 1 : k.find("Idx")]
                target = target[0].upper() + target[1:]
                if target not in collections:
                    problem = RuntimeError(
                        "Parsing indexer %s, expected to find collection %s but did not"
                        % (k, target)
                    )
                    if self.__class__.warn_missing_crossrefs:
                        warnings.warn(str(problem), RuntimeWarning)
                        continue
                    else:
                        raise problem
                branch_forms[k + "G"] = transforms.local2global_form(
                    branch_forms[k], branch_forms["o" + target]
                )

        # Create nested indexer from Idx1, Idx2, ... arrays
        for name, indexers in self.nested_items.items():
            if all(idx in branch_forms for idx in indexers):
                branch_forms[name] = transforms.nestedindex_form(
                    [branch_forms[idx] for idx in indexers]
                )

        # Create any special arrays
        for name, (fcn, args) in self.special_items.items():
            if all(k in branch_forms for k in args):
                branch_forms[name] = fcn(*(branch_forms[k] for k in args))

        output = {}
        for name in collections:
            mixin = self.mixins.get(name, "NanoCollection")
            if "o" + name in branch_forms and name not in branch_forms:
                # list collection
                offsets = branch_forms["o" + name]
                content = {
                    k[len(name) + 1 :]: branch_forms[k]
                    for k in branch_forms
                    if k.startswith(name + "_")
                }
                output[name] = zip_forms(
                    content, name, record_name=mixin, offsets=offsets
                )
                output[name]["content"]["parameters"].update(
                    {
                        "__doc__": offsets["parameters"]["__doc__"],
                        "collection_name": name,
                    }
                )
            elif "o" + name in branch_forms:
                # list singleton, can use branch's own offsets
                output[name] = branch_forms[name]
                output[name]["parameters"].update(
                    {"__array__": mixin, "collection_name": name}
                )
            elif name in branch_forms:
                # singleton
                output[name] = branch_forms[name]
            else:
                # simple collection
                output[name] = zip_forms(
                    {
                        k[len(name) + 1 :]: branch_forms[k]
                        for k in branch_forms
                        if k.startswith(name + "_")
                    },
                    name,
                    record_name=mixin,
                )
                output[name]["parameters"].update({"collection_name": name})

        return output

#parameters
#directory where the HNL data are stored
local_dir = '/Users/debryas/cernbox/HNL/skimmed_samples/nanoAOD/2018/HNL_tau/'
#directory where the triggers lumi file are stored
path_trigger = '/Users/debryas/Desktop/PhD_work/HNL_tau_analysis/accept/'
#directory where the figures are stored
path_fig = '/Users/debryas/Desktop/PhD_work/HNL_tau_analysis/HLT_analysis/figures/'

# Reco event selection: require at least 1 reco tau and 3 reco leptons (e/mu/tau)
#cut to apply
#tau
cut_tau_pt = 20. # Tau_pt > cut_tau_pt
cut_tau_eta = 2.3 #abs(Tau_eta) < cut_tau_eta
cut_tau_id = 2 # Tau_idDeepTau2017v2p1VSmu >= cut_tau_id; Tau_idDeepTau2017v2p1VSjet >= cut_tau_id; Tau_idDeepTau2017v2p1VSe >= cut_tau_id
#electrons
cut_e_pt = 5. # Electron_pt > cut_e_pt
cut_e_eta = 2.5 # abs(Electron_eta) < cut_e_eta
cut_e_id = 0 # (Electron_mvaFall17V2Iso_WP90 > cut_e_id || Electron_mvaFall17V2noIso_WP90 > cut_e_id)
#muons
cut_mu_pt = 5. # Muon_pt > cut_mu_pt
cut_mu_eta = 2.4 # abs(Muon_eta) < cut_mu_eta
cut_mu_id = 0 #(Muon_mediumId > cut_mu_id || Muon_tightId > cut_mu_id)
cut_mu_iso = 0.5 # Muon_pfRelIso03_all < cut_mu_iso

#mode = 'tte' #could be tte, ttm, tee, tmm, tem or all
not_wanted_HLT = ['Diphoton30_18_R9IdL_AND_HE_AND_IsoCaloId_NoPixelVeto','IsoTrackHB','MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_MET100']
trig_types = ['prompt']
eps = 0.01
eff_computation = 'channel_reco'

#save triggers in a variable
triggers = load_triggers(trig_types ,path_trigger)
print(str(len(triggers))+' HLT appears in this list (json file)')

#select color for plotting
color = ['cornflowerblue','indianred','sandybrown','darkseagreen','magenta']


#find the most common HLT for all mass by giving them a score
most_common_HLT_dic = {}
for mode in ['tte','ttm','tee','tmm','tem']:
    #load the file with optimal HLT
    with open('optimal_out/optimal_HLT_'+mode+'.json', 'r') as fp:
        HLT = json.load(fp)

    #remove bad HLT and find the list off all HLT available
    most_common_HLT_list = []
    score = []
    len_HLT = []
    i=0
    for HNLsamples in raw_signal_samples:
        most_common_HLT_list.append(HLT[HNLsamples])
        for bad_HLT in not_wanted_HLT:
            try:
                while True:
                    most_common_HLT_list[-1].remove(bad_HLT)
            except ValueError:
                pass
        len_HLT.append(len(most_common_HLT_list[-1]))
        score.append(1+np.array(range(len(most_common_HLT_list[-1]))))

    list = np.concatenate(most_common_HLT_list).tolist()
    HLT_found = pd.value_counts(np.array(list)).index.values

    #give them a score (the lowest the better)
    score_list = []
    for HLT in HLT_found:
        #print(HLT)
        HLT_score = 0
        for i in range(len(raw_signal_samples)):
            if any(score[i][np.array(most_common_HLT_list[i]) == HLT]):
                HLT_score= HLT_score + np.sum(score[i][np.array(most_common_HLT_list[i]) == HLT])
            else:
                HLT_score= HLT_score + max(len_HLT)
        #print(HLT_score)
        score_list.append(HLT_score)

    #reorder them
    ordered_list = []
    i=0
    while (i < len(score_list)):
        found_index = np.where(np.partition(score_list, i)[i] == score_list)[0]
        for j in range(len(found_index)):
            index = found_index[j]
            ordered_list.append(HLT_found[index])
        i = i + j + 1
    print('5 most common HLT for mode ' + mode)
    print(ordered_list[0:5])
    most_common_HLT_dic[mode] = ordered_list[0:5]

#select the most common HLT
#most_common_HLT_ttt = []
#most_common_HLT_tte = ['Ele32_WPTight_Gsf_L1DoubleEG', 'DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg', 'Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1', 'PFMET120_PFMHT120_IDTight_PFHT60','PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60']
#most_common_HLT_ttm = ['IsoMu24', 'DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg', 'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60', 'IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_CrossL1','Mu50']
#most_common_HLT_tee = ['Ele32_WPTight_Gsf_L1DoubleEG', 'Ele32_WPTight_Gsf', 'Ele23_Ele12_CaloIdL_TrackIdL_IsoVL', 'PFMET120_PFMHT120_IDTight_PFHT60','DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg']
#most_common_HLT_tmm = ['IsoMu24', 'Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8', 'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60', 'IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_CrossL1','DoubleMu3_DCA_PFMET50_PFMHT60']
#most_common_HLT_tem = ['IsoMu24', 'Ele32_WPTight_Gsf_L1DoubleEG', 'Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ', 'PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60','IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_CrossL1']



for mode in ['tte','ttm','tee','tmm','tem']:
    eff_HLT = []
    err_eff_HLT = []

    raw_eff_HLT = []
    raw_err_eff_HLT = []
    for HNLsamples in raw_signal_samples:
        print('----------------------------------------------------------------------')
        print('HLT analysis for '+ HNLsamples+' and channel '+ mode)

        #load events in the HNL file
        fname = local_dir+raw_signal_samples[HNLsamples]
        events = NanoEventsFactory.from_root(fname,schemaclass=HNLnanoAODschema).events()
        n_events_tot = len(events.GenPart)

        #mask at gen level
        t_mask, mu_mask, e_mask, t_e_mask, t_mu_mask, t_h_mask = lepton_mask_gen(events)

        #sanity check: exactly 3 leptons selected for the different mask
        if ((ak.sum(ak.flatten(e_mask))+ak.sum(ak.flatten(mu_mask)) + ak.sum(ak.flatten(t_e_mask)) + ak.sum(ak.flatten(t_mu_mask)) + ak.sum(ak.flatten(t_h_mask)))/n_events_tot) != 3:
            print('Not exactly 3 candidate selected in the mask')

        #mask at reco level
        mask_reco = reco_mask_cut(events, cut_tau_pt,cut_tau_eta,cut_tau_id,cut_e_pt,cut_e_eta,cut_e_id,cut_mu_pt,cut_mu_eta,cut_mu_id,cut_mu_iso)

        mode_list = ['ttt','tte','ttm','tee','tmm','tem']
        if mode not in mode_list:
            mode_mask = ak.ones_like(np.ones(n_events_tot,dtype=bool))
            print('Default mode: all event selected')
        else:
            if mode == 'ttt':
                mode_mask = (ak.sum(t_h_mask, axis =1) == 3)
                #print('Events in ' + mode + ': ' + str(ak.sum(mode_mask)/n_events_tot*100))
            if mode == 'tte':
                mode_mask = (ak.sum(t_h_mask, axis =1) == 2) & ( (ak.sum(t_e_mask, axis =1) == 1) | (ak.sum(e_mask, axis =1) == 1) )
                #print('Events in ' + mode + ': ' + str(ak.sum(mode_mask)/n_events_tot*100))
            if mode == 'ttm':
                mode_mask = (ak.sum(t_h_mask, axis =1) == 2) & ( (ak.sum(t_mu_mask, axis =1) == 1) | (ak.sum(mu_mask, axis =1) == 1) )
                #print('Events in ' + mode + ': ' + str(ak.sum(mode_mask)/n_events_tot*100))
            if mode == 'tee':
                mode_mask = (ak.sum(t_h_mask, axis =1) == 1) & ( ( (ak.sum(t_e_mask, axis =1) == 1) & (ak.sum(e_mask, axis =1) == 1) ) | (ak.sum(t_e_mask, axis =1) == 2) | (ak.sum(e_mask, axis =1) == 2) )
                #print('Events in ' + mode + ': ' + str(ak.sum(mode_mask)/n_events_tot*100))
            if mode == 'tmm':
                mode_mask = (ak.sum(t_h_mask, axis =1) == 1) & ( ( (ak.sum(t_mu_mask, axis =1) == 1) & (ak.sum(mu_mask, axis =1) == 1) ) | (ak.sum(t_mu_mask, axis =1) == 2) | (ak.sum(mu_mask, axis =1) == 2) )
                #print('Events in ' + mode + ': ' + str(ak.sum(mode_mask)/n_events_tot*100))
            if mode == 'tem':
                mode_mask = (ak.sum(t_h_mask, axis =1) == 1) & ( ( (ak.sum(e_mask, axis =1) == 1) & (ak.sum(mu_mask, axis =1) == 1) ) | ( (ak.sum(e_mask, axis =1) == 1) & (ak.sum(t_mu_mask, axis =1) == 1) ) | ( (ak.sum(t_e_mask, axis =1) == 1) & (ak.sum(mu_mask, axis =1) == 1) ) | ( (ak.sum(t_e_mask, axis =1) == 1) & (ak.sum(t_mu_mask, axis =1) == 1) ) )
                #print('Events in ' + mode + ': ' + str(ak.sum(mode_mask)/n_events_tot*100))

        most_common_HLT = most_common_HLT_dic[mode]
        mask_events_selected = mask_reco & mode_mask
        print('number of events after reco and channel selection: '+ str(ak.sum(mask_events_selected)))

        #look the common hlt in the root file and in the json, and store it in hlt_columns
        columns = sorted([ str(c) for c in events['HLT'].fields ])
        hlt_columns = [c for c in columns if c in triggers]

        # Compute max luminosity
        max_act_lumi = max(triggers[hlt_columns[i]]['act_lumi'] for i in np.arange(len(hlt_columns)))

        eff_computation_list = ['full', 'channel', 'reco', 'channel_reco']
        if eff_computation not in eff_computation_list:
            n_events = n_events_tot
            print('Default mode: all event selected for efficency computation')
        else:
            if eff_computation == 'full':
                n_events = n_events_tot
            if eff_computation == 'channel':
                n_events = ak.sum(mode_mask)
            if eff_computation == 'reco':
                n_events = ak.sum(mask_reco)
            if eff_computation == 'channel_reco':
                n_events = ak.sum(mask_events_selected)

        saved_eff, saved_err_eff = cumu_eff(events, most_common_HLT, triggers, max_act_lumi, mask_events_selected, n_events)
        raw_saved_eff, raw_saved_err_eff = raw_eff(events, most_common_HLT, triggers, max_act_lumi, mask_events_selected, n_events)

        eff_HLT.append(saved_eff)
        err_eff_HLT.append(saved_err_eff)
        raw_eff_HLT.append(raw_saved_eff)
        raw_err_eff_HLT.append(raw_saved_err_eff)

    plot_cumulative_eff_mass(path_fig, most_common_HLT, color, eff_HLT, err_eff_HLT, mode, eff_computation)
    plot_raw_eff_mass(path_fig, most_common_HLT, color, raw_eff_HLT, raw_err_eff_HLT, mode, eff_computation)