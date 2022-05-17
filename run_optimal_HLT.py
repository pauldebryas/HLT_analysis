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

from utils import load_triggers,finding_optimal_HLT,lepton_mask_gen, reco_mask_cut, plot_cumulative_eff_opt_HLT

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

mode = 'tte' #could be tte, ttm, tee, tmm, tem or all
trig_types = ['prompt']
eps = 0.01
eff_computation = 'channel_reco'

#save triggers in a variable
triggers = load_triggers(trig_types ,path_trigger)
print(str(len(triggers))+' HLT appears in this list (json file)')

for mode in ['ttt','tte','ttm','tee','tmm','tem']:
    HLT = {}
    for HNLsamples in raw_signal_samples:

        print('----------------------------------------------------------------------')
        print('HLT analysis for '+ HNLsamples+' and channel '+ mode)

        #load events in the HNL file
        fname = local_dir+raw_signal_samples[HNLsamples]
        events = NanoEventsFactory.from_root(fname,schemaclass=HNLnanoAODschema).events()
        n_events_tot = len(events.GenPart)

        print('number of events: '+ str(n_events_tot))

        #mask at gen level
        t_mask, mu_mask, e_mask, t_e_mask, t_mu_mask, t_h_mask = lepton_mask_gen(events)

        #sanity check: exactly 3 leptons selected for the different mask
        if ((ak.sum(ak.flatten(e_mask))+ak.sum(ak.flatten(mu_mask)) + ak.sum(ak.flatten(t_e_mask)) + ak.sum(ak.flatten(t_mu_mask)) + ak.sum(ak.flatten(t_h_mask)))/n_events_tot) != 3:
            print('not exactly 3 candidate selected in the mask')

        #mask at reco level
        mask_reco = reco_mask_cut(events, cut_tau_pt,cut_tau_eta,cut_tau_id,cut_e_pt,cut_e_eta,cut_e_id,cut_mu_pt,cut_mu_eta,cut_mu_id,cut_mu_iso)

        print('number of events after reco selection: '+ str(ak.sum(mask_reco)))

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

        print('number of events after gen selection (mode ' + mode + '): '+ str(ak.sum(mode_mask)))

        mask_events_selected = mask_reco & mode_mask
        print('number of events after reco and channel selection: '+ str(ak.sum(mask_events_selected)))

        #look the common hlt in the root file and in the json, and store it in hlt_columns
        columns = sorted([ str(c) for c in events['HLT'].fields ])
        hlt_columns = [c for c in columns if c in triggers]

        print(str(len(columns))+' HLT appear in the root data file')
        print(str(len(hlt_columns))+' HLT to be analysed (common with HLT list)')

        # Compute max luminosity
        max_act_lumi = max(triggers[hlt_columns[i]]['act_lumi'] for i in np.arange(len(hlt_columns)))
        print('Total luminosity: ' + str(max_act_lumi))

        # Compute luminosity ratio (to compute efficiency)
        lumi_ratio = []
        for i in np.arange(len(hlt_columns)):
            name = hlt_columns[i]
            lumi_ratio.append(triggers[name]['eff_lumi']/max_act_lumi)

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

        index_HLT_selected, saved_eff, saved_err_eff = finding_optimal_HLT(events, mask_events_selected, hlt_columns, eps, n_events, lumi_ratio)

        plot_cumulative_eff_opt_HLT(path_fig, HNLsamples, mode, hlt_columns, index_HLT_selected, saved_eff, saved_err_eff)

        #save HLT for each mass
        HLT_selected = []
        for i in range(len(index_HLT_selected)):
            HLT_selected.append(hlt_columns[index_HLT_selected[i]])

        HLT[HNLsamples] = HLT_selected

    #save 
    with open('optimal_out/optimal_HLT_'+mode+'.json', 'w') as fp:
        json.dump(HLT, fp)
