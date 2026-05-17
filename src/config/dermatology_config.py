CONFIG = {

    "name": "dermatology",

    "data_path": "data/dermatology/data.csv",

    "target": "class",

    # all are numerical features
    "num_cols": [

        "erythema",
        "scaling",
        "definite_borders",
        "itching",
        "koebner_phenomenon",
        "polygonal_papules",
        "follicular_papules",
        "oral_mucosal_involvement",
        "knee_and_elbow_involvement",
        "scalp_involvement",
        "family_history",
        "melanin_incontinence",
        "eosinophils_infiltrate",
        "PNL_infiltrate",
        "fibrosis_papillary_dermis",
        "exocytosis",
        "acanthosis",
        "hyperkeratosis",
        "parakeratosis",
        "clubbing_rete_ridges",
        "elongation_rete_ridges",
        "thinning_suprapapillary_epidermis",
        "spongiform_pustule",
        "munro_microabcess",
        "focal_hypergranulosis",
        "disappearance_granular_layer",
        "vacuolisation_damage_basal_layer",
        "spongiosis",
        "saw_tooth_appearance_retes",
        "follicular_horn_plug",
        "perifollicular_parakeratosis",
        "inflammatory_mononuclear_infiltrate",
        "band_like_infiltrate",
        "age"
    ],

    # no categorical columns
    "cat_cols": [],

    # PCA
    "pca_components": 10,

    # SVM
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",

    # split
    "test_size": 0.2,
    "stratify": True
}