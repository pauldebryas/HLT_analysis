import os
import re
import numpy as np
from import_signal_info import get_selected_sample_infos

def delta_r2(v1, v2):
    '''Calculates deltaR squared between two particles v1, v2 whose
    eta and phi methods return arrays
    '''
    dphi = (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi
    deta = v1.eta - v2.eta
    dr2 = dphi**2 + deta**2
    return dr2

def delta_r(v1, v2):
    '''Calculates deltaR between two particles v1, v2 whose
    eta and phi methods return arrays.
    
    Note: Prefer delta_r2 for cuts.
    '''
    return np.sqrt(delta_r2(v1, v2))

def delta_phi(v1, v2):
    return (v1.phi - v2.phi + np.pi) % (2 * np.pi) - np.pi

def cos_opening_angle(v, pv, p1, p2):
    '''Calculates cosine of opening angle between passed
    vertex v (with methods x, y, z), primary vertex pv,
    and the four-vector sum of two passed particles
    p1 and p2 (with methods pt/eta/phi)
    '''
    x = v.vtx_x - pv.x
    y = v.vtx_y - pv.y
    z = v.vtx_z - pv.z
    px = (p1.pt * np.cos(p1.phi) + p2.pt * np.cos(p2.phi))
    py = (p1.pt * np.sin(p1.phi) + p2.pt * np.sin(p2.phi))
    pz = (p1.pt * np.sinh(p1.eta) + p2.pt * np.sinh(p2.eta))
    
    num = x*px + y*py + z*pz
    den = np.sqrt(x**2 + y**2 + z**2) * np.sqrt(px**2 + py**2 + pz**2)

    return num/den

def dxy_significance(v, pv):
    dxy2 = (v.vtx_x - pv.x)**2 + (v.vtx_y - pv.y)**2
    edxy2 = v.vtx_ex**2 + v.vtx_ey**2 #PV has no but anyway negligible error
    return np.sqrt(dxy2/edxy2)

def inv_mass(p1, p2):
    '''Calculates invariant mass from 
    two input particles in pt/eta/phi
    representation, assuming zero mass 
    (so it works for track input)'''
    x = (p1.pt * np.cos(p1.phi) + p2.pt * np.cos(p2.phi))
    y = (p1.pt * np.sin(p1.phi) + p2.pt * np.sin(p2.phi))
    pz1 = p1.pt * np.sinh(p1.eta)
    pz2 = p2.pt * np.sinh(p2.eta)
    z = pz1 + pz2
    e = np.sqrt(p1.pt**2 + pz1**2) + np.sqrt(p2.pt**2 + pz2**2)
    e2 = e**2
    m2 = e2 - x**2 - y**2 - z**2
    return np.sqrt(m2)

def inv_mass_3p(p1, p2, p3):
    '''Calculates invariant mass from 
    three input particles in pt/eta/phi
    representation, assuming zero mass 
    (so it works for track input)'''
    x = p1.pt * np.cos(p1.phi) + p2.pt * np.cos(p2.phi) + p3.pt * np.cos(p3.phi)
    y = p1.pt * np.sin(p1.phi) + p2.pt * np.sin(p2.phi) + p3.pt * np.sin(p3.phi)
    z = p1.pt * np.sinh(p1.eta) + p2.pt * np.sinh(p2.eta) + p3.pt * np.sinh(p3.eta)
    e = np.sqrt(p1.pt**2 + (p1.pt * np.sinh(p1.eta))**2) + np.sqrt(p2.pt**2 + (p2.pt * np.sinh(p2.eta))**2) + np.sqrt(p3.pt**2 + (p3.pt * np.sinh(p3.eta))**2)
    m2 = e**2 - x**2 - y**2 - z**2
    return np.sqrt(m2)

def files_from_dir(d):
    '''Returns a list of all ROOT files in the passed directory.
    '''
    files = os.listdir(d)
    return ['/'.join([d, f]) for f in files if f.endswith('.root')]

def files_from_dirs(dirs):
    '''Returns a list of all ROOT files from the directories in the passed list.
    '''
    files = []
    for d in dirs:
        files += files_from_dir(d)
    return files

def get_info_from_name(sample_name):
    '''Returns sample info from HNL signal sample name.
    
    To be extended for non-mu/non-majorana/non-trilepton/non-2018 samples
    '''
    infos = get_selected_sample_infos(max_mass=11., process='majorana', lepton_type='tau', final_state='trilepton', year='2018')
    mass = float(re.search('_M-(.*)_V',sample_name).group(1))
    coupling = re.search('_V-(.*)_tau', sample_name).group(1).replace('_', '.')
    info = [info for info in infos if info.mass == mass and coupling in info.directory]
    if len(info) != 1:
        raise RuntimeError('Could not find unique cross section/ctau info for sample', 'sample_name', len(info))
    return info[0]
    
def xsec_from_name(sample_name):
    return get_info_from_name(sample_name).cross_section

def weight_to_new_ctau(old_ctau, old_v2, new_v2, ct):
    '''
    Returns an event weight based on the ratio of the normalised lifetime distributions.
    old_ctau: reference ctau
    old_v2  : reference coupling squared
    new_v2  : target coupling squared
    ct      : heavy neutrino lifetime in the specific event
    '''
    r = new_v2/old_v2
    weight = r * np.exp( ((1.-r)/old_ctau)*ct)

    return weight

cut_to_tex = {
    'all':'lumi $\times$ xsec', 
    'incoming':'Skimming (IsoMu24)', 
    'hnl_twomuon':'HNL two muons', 
    'hnl_twomuon_acc':'HNL two-muon acc.', 
    'lead_muon_id':'Leading muon ID', 
    'isomu24':'IsoMu24', 
    'dsapair':'DSA pair', 
    'dsapair_dsasel':'DSA selection',
    'muon_veto':'DSA glob/tr muon veto', 
    'muon_veto_tight':'DSA glob/tr muon veto tight', 
    'sta_match':'STA match', 
    'sta_match_unique':'Unique STA match',
    'muon_time':'STA time', 
    'muon_time_tight':'STA time tight', 
    'charge':'Same charge', 
    'zveto':'Z veto', 
    'delta_r':'$\Delta R$(DSA, DSA)', 
    'delta_phi':'$\Delta\phi$(DSA, prompt)', 
    'prob':'SV probability', 
    'dsapair_pt':'DSA pair $p_{T}$',
    'bjet_veto':'b jet veto  ',
    'opening_angle':'Opening angle',
    'wmass':'W mass veto',
    'pt_tight':'DSA $p_{T} > 5\,$GeV', 
    'pt_vtight':'DSA $p_{T} > 7\,$GeV',
    'dsa_jet_veto':'Jet veto',
    'sta_match_charge':'STA charge match', 
    'dxy_sig':'$d_\text{xy}$ sign.',
    'ch_iso':'DSA ch. iso$ < 2.5\,$GeV',

    'stapair_stasel':'STA pair',
    'stapair_muon_veto':'prompt veto',
    'stapair_time':'time selection',
    'stapair_charge':'Same charge',
    'stapair_zveto':'Z veto',
    'stapair_delta_r':'$\Delta R$(STA, STA)' ,
    'stapair_delta_phi':'$\Delta\phi$(STA, prompt)',
    'stapair_prob':'SV probability',
    'stapair_pt':'STA pair $p_{T}$',
    'stapair_bjet_veto':'b jet veto',
    'stapair_opening_angle':'Opening angle',
    'stapair_wmass':'W mass veto',
    'stapair_muon_veto_tight':'prompt veto tight',
    'stapair_pt_vtight':'$p_{T}$ > 7 GeV',
    'stapair_dsa_jet_veto':'Matched jet veto',
    'stapair_dxy_sig':'$d_\text{xy}$ sign.',
    
    'patdsa_patdsa':'PAT-DSA pair',
    'patdsa_stamatch':'DSA STA match',
    'patdsa_muon_time':'STA time',
    'patdsa_muon_time_tight':'STA time tight',
    'patdsa_bjet_veto':'b jet veto',
    'patdsa_prob':'SV probability',
    'patdsa_charge':'Same charge',
    'patdsa_dr':'$\Delta R$(DSA, PAT)',
    'patdsa_dphi':'$\Delta\phi$(DSA/PAT, prompt)',
    'patdsa_pt':'DSA-STA pair $p_{T}$',
    'patdsa_pt_dsa_tight':'DSA-STA DSA $p_{T} > 5\,$GeV',
    'patdsa_opening_angle':'Opening angle',
    'patdsa_wmass':'W mass',
    'patdsa_dxy_sig':'$d_\text{xy}$ sign.', 
    'patdsa_pt_tight':'DSA/PAT $p_{T} > 5\,$GeV',
    'patdsa_pat_iso':'PAT comb iso',
    'patdsa_ch_iso':'DSA charged iso',

    'patpat_patpat':'PAT-PAT pair',
    'patpat_bjet_veto':'b jet veto',
    'patpat_prob':'SV probability',
    'patpat_charge':'Same charge',
    'patpat_dr':'$\Delta R$(PAT, PAT)',
    'patpat_pt':'PAT pair $p_{T}$',
    'patpat_opening_angle':'Opening angle',
    'patpat_wmass':'W mass',
    'patpat_dxy_sig':'$d_\text{xy}$ sign.', 
    'patpat_pt_tight':'PAT $p_{T} > 5\,$GeV',
    'patpat_ch_iso':'PAT iso'
}