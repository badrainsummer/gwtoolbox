HE_detectors = {
    # Fermi/GBM
    "Fermi_GBM": {
        "Erange": [0.01, 1], # Energy range in MeV
        "sens": 2.28,#trigger threshold in ph/cm2/s
        "flux_unit": 'ph',
        "window": 64e-3, #width of photon/energy accumulation window, in s
        "bkg": .456, #ph/cm2/s
        "FoV": 0.6, # Field of View
        "corr": 2},
    # Swift/BAT
    "Swift_BAT": {
        "Erange": [0.015, 0.15], # Energy range
        "sens": 1.5,#0.9
        "flux_unit": 'ph',
        "window": 20*1e-3, #20 ms
        "bkg": .3, #ph/s/cm2
        "FoV": 0.1,
        "corr": 3},
    # Konus
    "Konus": {
        "Erange": [0.02, 15],
        "sens": 1e-7,
        "FoV": 0.8,
        "corr": 1},
    # insight/HE
    "insight_HE": {
        "Erange": [0.03, 0.1],
        "sens": 2e-8,
        "FoV": 0.6,
        "corr": 1},
    # GECAM
    "GECAM": {
        "Erange": [0.05, .4],#[0.006, 5]
        "sens": 2.0e-8,
        "flux_unit": 'ergs',
        "window": 50e-3,
        "bkg": 0.4e-8,
        "FoV": 1.,
        "corr": 3.5},
    # SVOM/GRM
    "SVOM_GRM": {
        "Erange": [0.03, 5],
        "sens": 2e-8,
        "FoV": 0.2,
        "corr": 1},
    # SVOM/ECLAIRs
    "SVOM_ECLAIRs": {
        "Erange": [0.004, 0.25],
        "sens": 2e-8,
        "FoV": 0.2,
        "corr": 1},
    # BATSE
    "BATSE": {
        "Erange": [0.05, .3],
        "sens": .3,#photons/cm2/s
        "flux_unit": 'ph',
        "window": 1,
        "bkg": .06, #ph/s/cm2
        "FoV": .25,
        "corr": 1},
}