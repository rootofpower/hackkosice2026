DATASET_ROOT = "dataset"

PAIRS = {
    "chory": {
        "disease_train": "dataset/chory/learning/chore",
        "disease_test":  "dataset/chory/test/chore",
        "healthy_train": "dataset/chory/learning/zdravi_ludi/super_proccesed",
        "healthy_test":  "dataset/chory/test/zdravi_ludi/super_proccesed",
        "label":         "Vsetky Choroby vs Zdravi",
    },
    "diabetes": {
        "disease_train": "dataset/diabetes/learning/diabetes",
        "disease_test":  "dataset/diabetes/test/diabetes",
        "healthy_train": "dataset/diabetes/learning/zdravi_ludi/super_proccesed",
        "healthy_test":  "dataset/diabetes/test/zdravi_ludi/super_proccesed",
        "label":         "Diabetes + Suche Oko",
    },
    "pgov": {
        "disease_train": "dataset/pgov/learning/pgov",
        "disease_test":  "dataset/pgov/test/pgov",
        "healthy_train": "dataset/pgov/learning/zdravi_ludi/super_proccesed",
        "healthy_test":  "dataset/pgov/test/zdravi_ludi/super_proccesed",
        "label":         "PGOV + Suche Oko",
    },
    "skleroza": {
        "disease_train": "dataset/skleroza/learning/skleroza",
        "disease_test":  "dataset/skleroza/test/skleroza",
        "healthy_train": "dataset/skleroza/learning/zdravi_ludi",
        "healthy_test":  "dataset/skleroza/test/zdravi_ludi",
        "label":         "Skleroza Multiplex",
    },
    "suche_oko": {
        "disease_train": "dataset/suche_oko/learning/super_proccesed",
        "disease_test":  "dataset/suche_oko/test/super_proccesed",
        "healthy_train": "dataset/suche_oko/learning/96_zdrave_ludi/96_ludi",
        "healthy_test":  "dataset/suche_oko/test/96_zdrave_ludi/96_ludi",
        "label":         "Suche Oko",
    },
}
