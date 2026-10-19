"""Shared definitions for the morphometric (FreeSurfer) tier."""

# Required features (spec): left/right kept separate. No age/sex by design.
VOLUME_FEATURES = [
    "fs_left_hippocampus_mm3", "fs_right_hippocampus_mm3",
    "fs_left_amygdala_mm3", "fs_right_amygdala_mm3",
    "fs_left_inf_lat_vent_mm3", "fs_right_inf_lat_vent_mm3",
]
THICKNESS_FEATURES = [  # raw ThickAvg, never eTIV-normalized
    "fs_lh_entorhinal_thickavg_mm", "fs_rh_entorhinal_thickavg_mm",
    "fs_lh_parahippocampal_thickavg_mm", "fs_rh_parahippocampal_thickavg_mm",
]
ETIV = "fs_etiv_mm3"
OPTIONAL_VOLUME_FEATURES = [
    "fs_left_lateral_ventricle_mm3", "fs_right_lateral_ventricle_mm3",
    "fs_cortexvol_mm3",
]
ALL_FS_COLUMNS = VOLUME_FEATURES + THICKNESS_FEATURES + [ETIV] + OPTIONAL_VOLUME_FEATURES
ID_COLUMNS = ["session_id", "subject_id", "supervised_label"]
